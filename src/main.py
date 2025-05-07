import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../data/IMDB Dataset.csv")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text) # remove html tags like <br /> etc
    text = re.sub(r"http\S+|www\S+", "", text) # remove URLs
    text = re.sub(r"[^a-z\s]", "", text) # remove punctuation (!?- etc) and numbers
    text = re.sub(r"\s+", " ", text).strip() # remove unnecesary whitespaces (fe "  ")
    return text


df["review"] = df["review"].apply(clean_text)


label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["sentiment"])  # positive=1, negative=0


X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["label"], test_size=0.2, stratify=df["label"]
)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english') # max features - we can try this out both ways to see the impact, same for ngrams

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


