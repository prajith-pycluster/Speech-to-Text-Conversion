import os
import wave
import threading
import pyaudio
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)

def record_audio(file_name="output.wav", channels=1, rate=44100, chunk=1024):
    """
    Records audio and saves it as a WAV file.
    """
    format = pyaudio.paInt16
    is_recording = True

    def stop_recording():
        nonlocal is_recording
        input("Press Enter to stop recording...")
        is_recording = False

    stop_thread = threading.Thread(target=stop_recording)
    stop_thread.start()

    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    print()
    logging.info("Recording started. Speak into the microphone.")
    frames = []

    while is_recording:
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    logging.info("Recording stopped.")
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    logging.info(f"Audio saved as {file_name}")


def transcribe_audio(path):
    """
    Transcribes audio using Google Speech Recognition.
    """
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio_listened = r.record(source)
        text = r.recognize_google(audio_listened)
    return text


def process_chunk(chunk, index, folder_name):
    """
    Processes a single audio chunk: saves and transcribes it.
    """
    chunk_filename = os.path.join(folder_name, f"chunk{index}.wav")
    chunk.export(chunk_filename, format="wav")
    try:
        text = transcribe_audio(chunk_filename)
        logging.info(f"Chunk {index} transcribed: {text}")
        return text
    except sr.UnknownValueError:
        logging.error(f"Chunk {index} could not be transcribed.")
        return ""


def get_large_audio_transcription_on_silence(path):
    """
    Splits audio into chunks and transcribes each chunk.
    """
    sound = AudioSegment.from_file(path)
    chunks = split_on_silence(sound,
                              min_silence_len=500,
                              silence_thresh=sound.dBFS - 14,
                              keep_silence=500)
    folder_name = "audio-chunks"
    os.makedirs(folder_name, exist_ok=True)

    def process_chunk_with_index(args):
        """
        Wrapper function for processing a chunk with index.
        """
        index, chunk = args
        return process_chunk(chunk, index, folder_name)

    # Using enumerate and mapping the chunks with indices
    chunk_with_indices = enumerate(chunks, start=1)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk_with_index, chunk_with_indices))

    # Cleanup temporary files
    for file in os.listdir(folder_name):
        os.remove(os.path.join(folder_name, file))
    os.rmdir(folder_name)

    return " ".join(results)


    
