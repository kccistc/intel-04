import pyaudio
import wave
import datetime
from pydub import AudioSegment
import sys

def convert_audio_to_16kHz(input_file, output_file):
    audio = AudioSegment.from_wav(input_file)
    audio = audio.set_frame_rate(16000)
    audio.export(output_file, format="wav")

def record_audio(filename, duration):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python Mic_in_wav_out.py <output_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    record_audio(filename, duration=10)  # Record for 20 seconds
    convert_audio_to_16kHz(filename, filename)
    print("Recording saved as:", filename)

if __name__ == "__main__":
    main()
