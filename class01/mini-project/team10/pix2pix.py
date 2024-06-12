if __name__ == "__main__":
    print("This is not an executable script. Please run main.py.")


import sys
import copy
import gc
from pathlib import Path
import requests
import types

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils import load_image
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
import numpy as np
import openvino as ov
from peft import LoraConfig
import torch
import torchvision.transforms.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from tqdm import tqdm
from PyQt5.QtGui import QImage, QPixmap
import cv2


# Sketch to Image 모델 경로
repo_dir = Path("img2img-turbo")

# 모델이 있는지 확인
if not repo_dir.exists():
    raise FileNotFoundError("img2img-turbo repository not found. Please clone the repository first.\nType 'git clone https://github.com/GaParmar/img2img-turbo.git' in terminal.")

# CPU를 사용하도록 코드 수정
pix2pix_turbo_py_path = repo_dir / "src/pix2pix_turbo.py"
model_py_path = repo_dir / "src/model.py"
orig_pix2pix_turbo_path = pix2pix_turbo_py_path.parent / ("orig_" + pix2pix_turbo_py_path.name)
orig_model_py_path = model_py_path.parent / ("orig_" + model_py_path.name)

if not orig_pix2pix_turbo_path.exists():
    pix2pix_turbo_py_path.rename(orig_pix2pix_turbo_path)

    with orig_pix2pix_turbo_path.open("r") as f:
        data = f.read()
        data = data.replace("cuda", "cpu")
        with pix2pix_turbo_py_path.open("w") as out_f:
            out_f.write(data)

if not orig_model_py_path.exists():
    model_py_path.rename(orig_model_py_path)

    with orig_model_py_path.open("r") as f:
        data = f.read()
        data = data.replace("cuda", "cpu")
        with model_py_path.open("w") as out_f:
            out_f.write(data)


# Pix2Pix 모듈 불러오기
sys.path.append('img2img-turbo/src')
from model import make_1step_sched
from pix2pix_turbo import TwinConv


tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")


def tokenize_prompt(prompt):
    caption_tokens = tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    return caption_tokens


def _vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    current_down_blocks = l_blocks
    return sample, current_down_blocks


def _vae_decoder_fwd(self, sample, incoming_skip_acts, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def vae_encode(self, x: torch.FloatTensor):
    """
    Encode a batch of images into latents.

    Args:
        x (`torch.FloatTensor`): Input batch of images.

    Returns:
        The latent representations of the encoded images. If `return_dict` is True, a
        [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
    """
    h, down_blocks = self.encoder(x)

    moments = self.quant_conv(h)
    posterior = DiagonalGaussianDistribution(moments)

    return (posterior, down_blocks)


def vae_decode(self, z: torch.FloatTensor, skip_acts):
    decoded = self._decode(z, skip_acts)[0]
    return (decoded,)


def vae__decode(self, z: torch.FloatTensor, skip_acts):
    z = self.post_quant_conv(z)
    dec = self.decoder(z, skip_acts)

    return (dec,)


class Pix2PixTurbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cpu()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = types.MethodType(_vae_encoder_fwd, vae.encoder)
        vae.decoder.forward = types.MethodType(_vae_decoder_fwd, vae.decoder)
        vae.encode = types.MethodType(vae_encode, vae)
        vae.decode = types.MethodType(vae_decode, vae)
        vae._decode = types.MethodType(vae__decode, vae)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cpu()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cpu()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cpu()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cpu()
        vae.decoder.ignore_skip = False
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        ckpt_folder = Path(ckpt_folder)

        if pretrained_name == "edge_to_image":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
            ckpt_folder.mkdir(exist_ok=True)
            outf = ckpt_folder / "edge_to_image_loras.pkl"
            if not outf:
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
                with open(outf, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name == "sketch_to_image_stochastic":
            # download from url
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
            ckpt_folder.mkdir(exist_ok=True)
            outf = ckpt_folder / "sketch_to_image_stochastic_lora.pkl"
            if not outf.exists():
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
                with open(outf, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                if k not in _sd_vae:
                    continue
                _sd_vae[k] = sd["state_dict_vae"][k]

            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cpu")
        vae.to("cpu")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cpu").long()
        self.text_encoder.requires_grad_(False)

    def set_r(self, r):
        self.unet.set_adapters(["default"], weights=[r])
        set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
        self.r = r
        self.unet.conv_in.r = r
        self.vae.decoder.gamma = r

    def forward(self, c_t, prompt_tokens, noise_map):
        caption_enc = self.text_encoder(prompt_tokens)[0]
        # scale the lora weights based on the r value
        sample, current_down_blocks = self.vae.encode(c_t)
        encoded_control = sample.sample() * self.vae.config.scaling_factor
        # combine the input and noise
        unet_input = encoded_control * self.r + noise_map * (1 - self.r)

        unet_output = self.unet(
            unet_input,
            self.timesteps,
            encoder_hidden_states=caption_enc,
        ).sample
        x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor, current_down_blocks)[0]).clamp(-1, 1)
        return output_image


def numpy_to_qpixmap(numpy_array):
    """
    Convert numpy array to QPixmap object.
    """
    height, width, channel = numpy_array.shape
    bytes_per_line = 3 * width
    qimage = QImage(numpy_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
    qpixmap = QPixmap.fromImage(qimage)
    return qpixmap


def sketch_to_image(sketch_path, keyword):
    # PyTorch 모델 불러오기
    ov_model_path = Path("models/pix2pix-turbo.xml")
    pt_model = None

    if not ov_model_path.exists():
        pt_model = Pix2PixTurbo("sketch_to_image_stochastic")
        pt_model.set_r(0.4)
        pt_model.eval()

    # OpenVINO 모델로 변환
    if not ov_model_path.exists():
        example_input = [torch.ones((1, 3, 512, 512)), torch.ones([1, 77], dtype=torch.int64), torch.ones([1, 4, 64, 64])]
        with torch.no_grad():
            ov_model = ov.convert_model(pt_model, example_input=example_input, input=[[1, 3, 512, 512], [1, 77], [1, 4, 64, 64]])
            ov.save_model(ov_model, ov_model_path)
        del ov_model
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()
    del pt_model
    gc.collect();

    # OpenVINO 초기화
    core = ov.Core()
    device = "CPU"

    # 모델 컴파일
    compiled_model = core.compile_model(ov_model_path, device)

    # 스케치 이미지 불러오기
    sketch_image = load_image(sketch_path)

    # 이미지 생성
    torch.manual_seed(145)
    c_t = torch.unsqueeze(F.to_tensor(sketch_image) > 0.5, 0)
    noise = torch.randn((1, 4, 512 // 8, 512 // 8))

    prompt_template = "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting"
    prompt = prompt_template.replace("{prompt}", keyword)

    prompt_tokens = tokenize_prompt(prompt)

    result = compiled_model([1 - c_t.to(torch.float32), prompt_tokens, noise])[0]

    image_tensor = (result[0] * 0.5 + 0.5) * 255
    image = np.transpose(image_tensor, (1, 2, 0)).astype(np.uint8)
    
    output_dir = Path("output")
    if not output_dir.exists():
        output_dir.mkdir()
    cv2.imwrite(output_dir / "sketch_to_image.jpg", image)