#!/bin/bash

python3 inter_face_dect.py

python3 sound.py

ffmpeg -i ./output.wav -acodec pcm_s16le -ac 1 -ar 16000 ./out.wav -y

python3 speech_recognition_wav2vec_demo.py \
        -m ./public/wav2vec2-base/FP16/wav2vec2-base.xml \
        -i ./out.wav
