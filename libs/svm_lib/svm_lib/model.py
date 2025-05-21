from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from .preprocessing import clean_text
import joblib

def build_pipeline():
    return Pipeline([
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

def train(X_train, y_train):
    model = build_pipeline()
    model.fit(X_train, y_train)
    return model

def predict(model, texts):
    if isinstance(texts, str):
        texts = [texts]

    texts = [clean_text(text) for text in texts]

    return model.predict(texts)

def save_model(model, path="sentiment_model.pkl"):
    joblib.dump(model, path)

def load_model(path="sentiment_model.pkl"):
    return joblib.load(path)
