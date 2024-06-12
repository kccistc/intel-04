import os
import time
import datetime

def record_audio(iteration):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"recording_{iteration}_{current_time}.wav"
    os.system(f"python3 Mic_in_wav_out.py {filename}")
    return filename

def convert_wav_to_txt(wav_file, iteration):
    input_txt = f"input_{iteration}.txt"
    os.system(f"python3 speech_recognition_wav2vec_demo.py --model=public/wav2vec2-base/FP16/wav2vec2-base.xml -i {wav_file} --output={input_txt}")
    return input_txt

def perform_gpt2_text_prediction(input_txt, iteration):
    output_txt = f"output_{iteration}.txt"
    os.system(f"python3 gpt2_text_prediction_demo.py --model=public/gpt-2/FP16/gpt-2.xml --vocab=public/gpt-2/gpt2/vocab.json --merges=public/gpt-2/gpt2/merges.txt --input_file={input_txt} --output_file={output_txt}")
    return output_txt

def convert_txt_to_speech(output_txt, iteration):
    audio_file = f"audio_{iteration}.wav"
    os.system(f"python3 text_to_speech_demo.py -i {output_txt} -o {audio_file} -s_id 19 -m_duration intel/text-to-speech-en-multi-0001/text-to-speech-en-multi-0001-duration-prediction/FP16/text-to-speech-en-multi-0001-duration-prediction.xml -m_forward intel/text-to-speech-en-multi-0001/text-to-speech-en-multi-0001-regression/FP16/text-to-speech-en-multi-0001-regression.xml -m_melgan intel/text-to-speech-en-multi-0001/text-to-speech-en-multi-0001-generation/FP16/text-to-speech-en-multi-0001-generation.xml")
    return audio_file

def main():
    for i in range(2):  # 이 과정을 5번 반복합니다.
        print(f"Iteration {i+1} started.")
        
        # 1. 마이크 입력을 .wav 파일로 저장
        wav_file = record_audio(i+1)
        
        # 2. .wav 파일을 .txt 파일로 변환
        input_txt = convert_wav_to_txt(wav_file, i+1)
        
        # 3. GPT-2 Text Prediction 실행
        output_txt = perform_gpt2_text_prediction(input_txt, i+1)
        
        # 4. Text-to-speech 실행
        audio_file = convert_txt_to_speech(output_txt, i+1)
        
        print(f"Iteration {i+1} completed.")
        print("Waiting for 5 minutes before next iteration...")
        time.sleep(20)  # 5분 대기

if __name__ == "__main__":
    main()
