import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset, DatasetDict

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from transformers.trainer_utils import EvalPrediction

# 1. Load the DataFrame
data = pd.read_csv("preprocessed_sentiment_data.csv")
df = pd.DataFrame(data)
print(df.head())

# Map labels to IDs (binary: positive=1, negative=0)
label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}

df["label_id"] = df["sentiment"].map(label2id)

# 2. Train/Val/Test Split
train_df, test_df = train_test_split(
    df, test_size=0.1, stratify=df["label_id"], random_state=42
)
train_df, val_df = train_test_split(
    train_df, test_size=0.1111, stratify=train_df["label_id"], random_state=42
)  # 80/10/10 split

# Convert pandas → Hugging Face Dataset
def to_hf(d: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        d[["review", "label_id"]].rename(columns={"review": "text", "label_id": "labels"}),
        preserve_index=False,
    )

raw_datasets = DatasetDict({
    "train": to_hf(train_df),
    "validation": to_hf(val_df),
    "test": to_hf(test_df),
})

# 3. Tokenizer
model_name = "distilbert-base-uncased"  # lightweight & good for binary
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

tokenized = raw_datasets.map(tokenize, batched=True, remove_columns=["text"])
# No need to set format, Trainer will handle conversion
pass


# 4. Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# 5. Metrics
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# 6. Training

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./binary_sentiment_model",
    eval_strategy="epoch",              
    save_strategy="epoch",            
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,       
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="./logs",
    logging_strategy="epoch"           
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# 7. Evaluation
preds = trainer.predict(tokenized["test"])
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["negative", "positive"], zero_division=0))

print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
