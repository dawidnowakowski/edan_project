import pandas as pd
import re
import argparse
import joblib
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation and digits
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


def train_model(input_csv_path, output_model_path):
    df = pd.read_csv(input_csv_path)
    df["review"] = df["review"].apply(clean_text)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["sentiment"])

    X_train, _, y_train, _ = train_test_split(
        df["review"], df["label"], test_size=0.2, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LinearSVC(C=0.1))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump((pipeline, label_encoder), output_model_path+"/svm.joblib")
    print(f"Model saved to {output_model_path}/svm.joblib")


def test_model(review_file_path, model_path):
    try:
        pipeline, label_encoder = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    try:
        with open(review_file_path, "r", encoding="utf-8") as file:
            review = file.read()
    except Exception as e:
        print(f"Error reading review file: {e}")
        sys.exit(1)

    cleaned_review = clean_text(review)
    prediction = pipeline.predict([cleaned_review])[0]
    sentiment = label_encoder.inverse_transform([prediction])[0]
    print(sentiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDB Review Sentiment Analysis")

    parser.add_argument("-t", "--task", choices=["training", "test"], required=True, help="Task type: training or test")
    parser.add_argument("-f", "--file", required=True, help="Input file path (CSV for training or TXT for test)")
    parser.add_argument("-m", "--model", required=True, help="Model directory path to save | file path to load")

    args = parser.parse_args()

    if args.task == "training":
        train_model(args.file, args.model)
    elif args.task == "test":
        test_model(args.file, args.model)
