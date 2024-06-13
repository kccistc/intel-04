import os
from tts_module import text_to_speech
from gpt_api import summarize_and_save_text
import subprocess
import time
import tkinter as tk
from tkinter import filedialog

def get_valid_file():
    valid_extensions = ['.jpg', '.png']  # 허용할 파일 확장자 목록

    file_path = filedialog.askopenfilename()
    if file_path:
        _, ext = os.path.splitext(file_path)
        if ext.lower() in valid_extensions:
            return file_path
        else:
            tk.messagebox.showerror("Error", "JPG 또는 PNG 파일만 선택할 수 있습니다.")
            return get_valid_file()  # 다시 파일 선택하도록 함수 재귀 호출
    else:
        return None

def inp_image():
    root = tk.Tk()
    root.withdraw()  # GUI 창 숨기기

    file_path = get_valid_file()
    if file_path:
        print("선택된 파일:", file_path)
    else:
        print("파일 선택이 취소되었습니다.")
    
    return file_path


def run_text_spotting_demo(input_image):
    mask_rcnn_model = "intel/text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml"
    text_enc_model = "intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml"
    text_dec_model = "intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16/text-spotting-0005-recognizer-decoder.xml"
    input_source = input_image  # Use 0 for webcam, or replace with image/video file path
    
    command = [
        "python3", "text_spotting_demo.py",
        "-m_m", mask_rcnn_model,
        "-m_te", text_enc_model,
        "-m_td", text_dec_model,
        "-i", input_source,
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print("Output:", result.stdout)
    print("Error:", result.stderr)


def main():
    tts_model_duration = 'model/text-to-speech-en-0001/text-to-speech-en-0001-duration-prediction/FP16/text-to-speech-en-0001-duration-prediction.xml'
    tts_model_forward = 'model/text-to-speech-en-0001/text-to-speech-en-0001-regression/FP16/text-to-speech-en-0001-regression.xml'
    tts_model_melgan = 'model/text-to-speech-en-0001/text-to-speech-en-0001-generation/FP16/text-to-speech-en-0001-generation.xml'    
    device = 'CPU'
    
    # 파일 입력
    input_image = inp_image()

    # OCR 수행
    run_text_spotting_demo(input_image)
    time.sleep(1)
    print("OCR 수행 완료!(ocr_results.txt)")

    # Summarize 수행
    answer = summarize_and_save_text('ocr_results.txt')
    text_path = answer
    print("GPT3.5를 통한 내용 요약 수행 완료!")
    print(f"요약 결과 : {answer}")

    # TTS 수행
    text_to_speech(text_path, tts_model_duration, tts_model_forward, tts_model_melgan, device)
    print("TTS 수행 완료!(output.wav)")

if __name__ == '__main__':
    main()