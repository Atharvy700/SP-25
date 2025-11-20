#bert-finetuned

import torch
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


data_files = {
        "train":      "trans_eq_data/train.json",
        "validation": "trans_eq_data/valid.json",
        "test":       "trans_eq_data/test.json",
}
ds = load_dataset("json", data_files=data_files)
test_data = ds["test"]

# 2) Mapping
label_to_id = {"no": 0, "yes": 1}
id_to_label = {v: k for k, v in label_to_id.items()}

label_names = ["no", "yes"]


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokenized["labels"] = [label_to_id[l] for l in examples["label"]]
    return tokenized


tokenized_ds = ds.map(preprocess_function, batched=True)
tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_ds = tokenized_ds["train"]
val_ds   = tokenized_ds["validation"]
test_ds  = tokenized_ds["test"]


model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-uncased",
    num_labels=len(label_names),
)

# Accuracy metric
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# Training hyper‑parameters
training_args = TrainingArguments(
    output_dir="finetuned_bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ---------------------------
# Evaluate on the test split
# ---------------------------
test_results = trainer.evaluate(eval_dataset=test_ds)
print(f"Fine‑tuned accuracy on test set: {test_results['eval_accuracy']:.4f}")