# train_binary_sentiment.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset, DatasetDict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import EvalPrediction

# ----------------------------
# 1️⃣ CPU device
# ----------------------------
device = "cpu"
print(f"Using device: {device}")

# ----------------------------
# 2️⃣ Load Dataset
# ----------------------------
data = pd.read_csv("preprocessed_sentiment_data.csv")
df = pd.DataFrame(data)
print(df.head())

label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}
df["label_id"] = df["sentiment"].map(label2id)

# ----------------------------
# 3️⃣ Train/Validation/Test Split
# ----------------------------
train_df, test_df = train_test_split(
    df, test_size=0.1, stratify=df["label_id"], random_state=42
)
train_df, val_df = train_test_split(
    train_df, test_size=0.1111, stratify=train_df["label_id"], random_state=42
)  # 80/10/10 split

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

# ----------------------------
# 4️⃣ Tokenizer
# ----------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
   
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized = raw_datasets.map(tokenize, batched=True, remove_columns=["text"])

# ----------------------------
# 5️⃣ Model
# ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
).to(device)

# ----------------------------
# 6️⃣ Metrics
# ----------------------------
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ----------------------------
# 7️⃣ Training Arguments
# ----------------------------
output_dir = "./binary_sentiment_model_cpu"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",       
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=50,          
    report_to="none",
)

# ----------------------------
# 8️⃣ Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ----------------------------
# 9️⃣ Resume from last checkpoint
# ----------------------------
last_checkpoint = None
if os.path.exists(output_dir):
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint")
    ]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]
        print(f"Resuming from last checkpoint: {last_checkpoint}")

trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)

# ----------------------------
# 🔟 Evaluation
# ----------------------------
preds = trainer.predict(tokenized["test"])
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

print("\nClassification Report:\n", classification_report(
    y_true, y_pred, target_names=["negative", "positive"], zero_division=0
))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

# ----------------------------
# 1️⃣1️⃣ Save Final Model
# ----------------------------
trainer.save_model("./final_model_cpu")
tokenizer.save_pretrained("./final_model_cpu")


print("Training completed. Model saved in './final_model_cpu'")
