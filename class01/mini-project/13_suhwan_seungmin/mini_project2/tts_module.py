import numpy as np
import wave
from tqdm import tqdm
from openvino import Core
from models.forward_tacotron_ie import ForwardTacotronIE
from models.mel2wave_ie import WaveRNNIE, MelGANIE

def save_wav(x, path):
    sr = 22050
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(x.tobytes())

def text_to_speech(text, model_duration, model_forward, model_melgan, device='CPU'):
    core = Core()
    vocoder = MelGANIE(model_melgan, core, device=device)
    forward_tacotron = ForwardTacotronIE(model_duration, model_forward, core, device, verbose=False)

    audio_res = np.array([], dtype=np.int16)
    len_th = 80
    texts = [text[i:i+len_th] for i in range(0, len(text), len_th)]

    for text in tqdm(texts):
        mel = forward_tacotron.forward(text)
        audio = vocoder.forward(mel)
        audio_res = np.append(audio_res, audio)

    save_wav(audio_res, 'output.wav')
