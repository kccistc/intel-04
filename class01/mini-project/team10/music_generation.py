from collections import namedtuple
from functools import partial
import gc
from pathlib import Path
from typing import Optional, Tuple
import warnings
import openvino as ov
import numpy as np
import torch
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
from torch.jit import TracerWarning
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)

# Ignore tracing warnings
warnings.filterwarnings("ignore", category=TracerWarning)

import sys
from packaging.version import parse


if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
loading_kwargs = {}

if parse(importlib_metadata.version("transformers")) >= parse("4.40.0"):
    loading_kwargs["attn_implementation"] = "eager"


models_dir = Path("./models")
t5_ir_path = models_dir / "t5.xml"
musicgen_0_ir_path = models_dir / "mg_0.xml"
musicgen_ir_path = models_dir / "mg.xml"
audio_decoder_ir_path = models_dir / "encodec.xml"


class MusicGenerator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.n_tokens = None
        self.sampling_rate = None

        self.init_model()
        self.convert_model_to_openvino()
    
    def init_model(self):
        # Load the pipeline
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", torchscript=True, return_dict=False, **loading_kwargs)

        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

        sample_length = 8 # seconds

        self.n_tokens = sample_length * self.model.config.audio_encoder.frame_rate + 3
        self.sampling_rate = self.model.config.audio_encoder.sampling_rate
        print("Sampling rate is", self.sampling_rate, "Hz")

        self.model.to("cpu")
        self.model.eval();

    def convert_model_to_openvino(self):
        sample_inputs = self.processor(
            text=["A song that would fit in a 1970s western. write me a guitar"],
            return_tensors="pt",
        )

        if not t5_ir_path.exists():
            t5_ov = ov.convert_model(self.model.text_encoder, example_input={"input_ids": sample_inputs["input_ids"]})

            ov.save_model(t5_ov, t5_ir_path)
            del t5_ov
            gc.collect()

        # Set model config `torchscript` to True, so the model returns a tuple as output
        self.model.decoder.config.torchscript = True

        if not musicgen_0_ir_path.exists():
            decoder_input = {
                "input_ids": torch.ones(8, 1, dtype=torch.int64),
                "encoder_hidden_states": torch.ones(2, 12, 1024, dtype=torch.float32),
                "encoder_attention_mask": torch.ones(2, 12, dtype=torch.int64),
            }
            mg_ov_0_step = ov.convert_model(self.model.decoder, example_input=decoder_input)

            ov.save_model(mg_ov_0_step, musicgen_0_ir_path)
            del mg_ov_0_step
            gc.collect()

        # Add additional argument to the example_input dict
        if not musicgen_ir_path.exists():
            # Add `past_key_values` to the converted model signature
            decoder_input["past_key_values"] = tuple(
                [
                    (
                        torch.ones(2, 16, 1, 64, dtype=torch.float32),
                        torch.ones(2, 16, 1, 64, dtype=torch.float32),
                        torch.ones(2, 16, 12, 64, dtype=torch.float32),
                        torch.ones(2, 16, 12, 64, dtype=torch.float32),
                    )
                ]
                * 24
            )

            mg_ov = ov.convert_model(self.model.decoder, example_input=decoder_input)
            for input in mg_ov.inputs[3:]:
                input.get_node().set_partial_shape(ov.PartialShape([-1, 16, -1, 64]))
                input.get_node().set_element_type(ov.Type.f32)

            mg_ov.validate_nodes_and_infer_types()

            ov.save_model(mg_ov, musicgen_ir_path)
            del mg_ov
            gc.collect()

        if not audio_decoder_ir_path.exists():

            class AudioDecoder(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, output_ids):
                    return self.model.decode(output_ids, [None])

            audio_decoder_input = {"output_ids": torch.ones((1, 1, 4, self.n_tokens - 3), dtype=torch.int64)}

            with torch.no_grad():
                audio_decoder_ov = ov.convert_model(AudioDecoder(self.model.audio_encoder), example_input=audio_decoder_input)
            ov.save_model(audio_decoder_ov, audio_decoder_ir_path)
            del audio_decoder_ov
            gc.collect()
        
    def generate_music(self, keyword):
        inputs = self.processor(
            text=[keyword],
            return_tensors="pt",
        )

        text_encode_ov = TextEncoderWrapper(t5_ir_path, self.model.text_encoder.config)
        musicgen_decoder_ov = MusicGenWrapper(
            musicgen_0_ir_path,
            musicgen_ir_path,
            self.model.decoder.config,
            self.model.decoder.num_codebooks,
            self.model.decoder.build_delay_pattern_mask,
            self.model.decoder.apply_delay_pattern_mask,
        )
        audio_encoder_ov = AudioDecoderWrapper(audio_decoder_ir_path, self.model.audio_encoder.config)

        del self.model.text_encoder
        del self.model.decoder
        del self.model.audio_encoder
        gc.collect()

        self.model.text_encoder = text_encode_ov
        self.model.decoder = musicgen_decoder_ov
        self.model.audio_encoder = audio_encoder_ov

        self.model.prepare_inputs_for_generation = partial(prepare_inputs_for_generation, self.model)

        print("Audio Generating")
        audio_values = self.model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=self.n_tokens)
        print("Complete")

        print(audio_values.shape)

        # 음악 파일 경로 및 파일 이름 설정
        output_audio_path = "output/generated_audio.wav"

        # audio_values의 첫 번째 샘플을 저장합니다.
        sf.write(output_audio_path, audio_values[0, 0].cpu().numpy(), self.sampling_rate)

        return output_audio_path


core = ov.Core()
device = "CPU"


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder_ir, config):
        super().__init__()
        self.encoder = core.compile_model(encoder_ir, device)
        self.config = config

    def forward(self, input_ids, **kwargs):
        last_hidden_state = self.encoder(input_ids)[self.encoder.outputs[0]]
        last_hidden_state = torch.tensor(last_hidden_state)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state)


class MusicGenWrapper(torch.nn.Module):
    def __init__(
        self,
        music_gen_lm_0_ir,
        music_gen_lm_ir,
        config,
        num_codebooks,
        build_delay_pattern_mask,
        apply_delay_pattern_mask,
    ):
        super().__init__()
        self.music_gen_lm_0 = core.compile_model(music_gen_lm_0_ir, device)
        self.music_gen_lm = core.compile_model(music_gen_lm_ir, device)
        self.config = config
        self.num_codebooks = num_codebooks
        self.build_delay_pattern_mask = build_delay_pattern_mask
        self.apply_delay_pattern_mask = apply_delay_pattern_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_attention_mask: torch.LongTensor = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        **kwargs
    ):
        if past_key_values is None:
            model = self.music_gen_lm_0
            arguments = (input_ids, encoder_hidden_states, encoder_attention_mask)
        else:
            model = self.music_gen_lm
            arguments = (
                input_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                *past_key_values,
            )

        output = model(arguments)
        return CausalLMOutputWithCrossAttentions(
            logits=torch.tensor(output[model.outputs[0]]),
            past_key_values=tuple([output[model.outputs[i]] for i in range(1, 97)]),
        )


class AudioDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder_ir, config):
        super().__init__()
        self.decoder = core.compile_model(decoder_ir, device)
        self.config = config
        self.output_type = namedtuple("AudioDecoderOutput", ["audio_values"])

    def decode(self, output_ids, audio_scales):
        output = self.decoder(output_ids)[self.decoder.outputs[0]]
        return self.output_type(audio_values=torch.tensor(output))


def prepare_inputs_for_generation(
    self,
    decoder_input_ids,
    past_key_values=None,
    attention_mask=None,
    head_mask=None,
    decoder_attention_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
    use_cache=None,
    encoder_outputs=None,
    decoder_delay_pattern_mask=None,
    guidance_scale=None,
    **kwargs,
):
    if decoder_delay_pattern_mask is None:
        (
            decoder_input_ids,
            decoder_delay_pattern_mask,
        ) = self.decoder.build_delay_pattern_mask(
            decoder_input_ids,
            self.generation_config.pad_token_id,
            max_length=self.generation_config.max_length,
        )

    # apply the delay pattern mask
    decoder_input_ids = self.decoder.apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask)

    if guidance_scale is not None and guidance_scale > 1:
        # for classifier free guidance we need to replicate the decoder args across the batch dim (we'll split these
        # before sampling)
        decoder_input_ids = decoder_input_ids.repeat((2, 1))
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat((2, 1))

    if past_key_values is not None:
        # cut decoder_input_ids if past is used
        decoder_input_ids = decoder_input_ids[:, -1:]

    return {
        "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        "encoder_outputs": encoder_outputs,
        "past_key_values": past_key_values,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
        "use_cache": use_cache,
    }


