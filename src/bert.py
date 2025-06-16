import argparse
import pandas as pd
import re
import torch
import joblib
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(example, tokenizer):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)


def train_bert(csv_path, model_output_path=None):
    df = pd.read_csv(csv_path)
    df["review"] = df["review"].apply(clean_text)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["sentiment"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["label"], test_size=0.2, stratify=df["label"]
    )

    train_dataset = Dataset.from_pandas(pd.DataFrame({"text": X_train, "label": y_train}))
    test_dataset = Dataset.from_pandas(pd.DataFrame({"text": X_test, "label": y_test}))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = train_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./tmp_ignore",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    trainer.evaluate()

    if model_output_path:
        model.save_pretrained(model_output_path)
        tokenizer.save_pretrained(model_output_path)
        joblib.dump(label_encoder, f"{model_output_path}/label_encoder.pkl")
        print(f"Model and tokenizer saved to {model_output_path}")


def test_bert(txt_path, model_path=None):
    with open(txt_path, "r", encoding="utf-8") as f:
        review = f.read()

    cleaned = clean_text(review)
 
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")

    tokens = tokenizer(cleaned, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    model.eval()

    with torch.no_grad():
        output = model(**tokens)
        prediction = torch.argmax(output.logits, dim=1).item()
        sentiment = label_encoder.inverse_transform([prediction])[0]

    print(sentiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDB Sentiment Classification with BERT")

    parser.add_argument("-t", "--task", choices=["training", "test"], required=True, help="Task type: training or test")
    parser.add_argument("-f", "--file", required=True, help="CSV file (for training) or TXT file (for test)")
    parser.add_argument("-m", "--model", required=True, help="Directory path to save|load model")

    args = parser.parse_args()

    if args.task == "training":
        train_bert(args.file, args.model)
    elif args.task == "test":
        test_bert(args.file, args.model)
