from audio_module import record_audio, get_large_audio_transcription_on_silence
from text_analysis import tokenize, remove_stopwords, sentiment_analysis_with_textblob, create_grouped_frequency_dataframe
import pyttsx3
import datetime

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def wish_me():
    """
    Greets the user based on the current time.
    """
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning!")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("Hello! This project converts spoken words into written text using advanced speech recognition technology.")
    speak('Recording will start in a few seconds...')

if __name__ == "__main__":
    # Greet the user
    wish_me()

    # Record audio
    record_audio(file_name="output.wav")

    # Transcribe audio
    path = "output.wav"
    full_text = get_large_audio_transcription_on_silence(path)
    print("\nFull Text:", full_text)

    # Text Analysis
    tokens = tokenize(full_text)
    filtered_tokens = remove_stopwords(tokens)
    freq_df = create_grouped_frequency_dataframe(filtered_tokens)
    sentiment = sentiment_analysis_with_textblob(full_text)

    # Print Results
    print("\nFiltered Tokens:", filtered_tokens)
    print("\nWord Frequencies:", freq_df)
    print("\nSentiment Analysis:", sentiment)

    # Speak Results
    speak(f"The sentiment analysis of your transcription is: {sentiment}")
