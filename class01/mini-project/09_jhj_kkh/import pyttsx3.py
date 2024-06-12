from gtts import gTTS
import os

def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_file)
    
# 예제 실행
text_to_speech("Hello, this is a test.", "output.mp3")