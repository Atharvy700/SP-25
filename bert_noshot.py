#bert no shot
import torch
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


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


classifier = pipeline(
    "zero-shot-classification",
    model="google-bert/bert-base-uncased",
    tokenizer="google-bert/bert-base-uncased",

)

# 4. Run zero-shot inference and collect predictions
preds = []
for ex in test_data:
    out = classifier(ex["text"], candidate_labels=label_names)
    preds.append(out["labels"][0])  # top predicted label

# 5. Compute accuracy
gold = test_data["label"]
accuracy = np.mean([p == g for p, g in zip(preds, gold)])
print(f"Zero-shot accuracy on test set: {accuracy:.4f}")