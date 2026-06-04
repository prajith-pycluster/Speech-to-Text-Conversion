# Audio Transcription & Text Analytics Pipeline

An advanced Python application that records real-time audio (or processes existing files), splits the audio on moments of silence for efficient multi-threaded transcription, and performs deep text analytics—including word frequency tracking and sentiment analysis.

---

## 🚀 Key Features

* **Asynchronous Audio Recording:** Captures high-quality live microphone input utilizing multi-threading so you can press 'Enter' to stop seamlessly.
* **Intelligent Audio Chunking:** Uses silence-detection thresholds to split massive audio files into clean, sentence-level blocks.
* **Concurrent Speech Recognition:** Leverages a `ThreadPoolExecutor` to send audio chunks to Google Speech Recognition in parallel, radically speeding up total processing time.
* **Natural Language Processing (NLP):** Auto-cleans transcripts by removing generic English stopwords to reveal core content.
* **Frequency & Sentiment Analytics:** Generates structured Pandas DataFrames classifying words by usage metrics and detects emotional tone (Positive/Negative/Neutral).

---

## 🛠️ Prerequisites & Installation

Make sure your system has **Python 3.8+** installed. You will also need **FFmpeg** installed on your system path for `pydub` to handle audio conversions.

### 1. Install Required Libraries

Install all external dependencies directly from your terminal:

```bash
pip install pyaudio SpeechRecognition pydub textblob nltk pandas
