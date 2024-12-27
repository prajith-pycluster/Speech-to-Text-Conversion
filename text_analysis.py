import re
from collections import Counter
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
import pandas as pd

nltk.download('stopwords')
nltk.download('vader_lexicon')

stop_words = set(stopwords.words('english'))

def tokenize(text):
    """
    Tokenizes the text into words.
    """
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words


def remove_stopwords(tokens):
    """
    Removes stopwords from the list of tokens.
    """
    return [word for word in tokens if word not in stop_words]


def word_frequency(tokens):
    """
    Calculates word frequency.
    """
    return Counter(tokens)


def create_grouped_frequency_dataframe(tokens):
    """
    Creates a Pandas DataFrame with words grouped by their frequency.
    """
    freq = word_frequency(tokens)
    freq_dict = {}
    for word, count in freq.items():
        if count not in freq_dict:
            freq_dict[count] = []
        freq_dict[count].append(word)
    
    # Convert to DataFrame
    df = pd.DataFrame(list(freq_dict.items()), columns=["Frequency", "Words"])
    df["Words"] = df["Words"].apply(", ".join)  # Combine words into a single string
    df = df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
    return df


def sentiment_analysis_with_textblob(text):
    """
    Performs sentiment analysis using TextBlob.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"
