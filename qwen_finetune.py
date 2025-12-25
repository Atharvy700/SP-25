#!/usr/bin/env python3
# qwen_finetune.py
import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

def load_json_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(example_text: str) -> str:
    return (
        "You are a careful reasoning assistant.\n"
        "Task: Answer the final question with exactly one word: yes or no.\n\n"
        f"{example_text}\n\n"
        "Answer (yes/no):"
    )

class YesNoSFTDataset(Dataset):
    """
    Causal-LM fine-tuning:
    input = prompt + answer
    labels mask the prompt tokens (-100), so loss trains only on the answer tokens.
    """
    def __init__(self, examples, tokenizer, max_length=2048):
        self.examples = examples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = build_prompt(ex["text"])
        answer = ex["label"].strip().lower()
        if answer not in {"yes", "no"}:
            answer = "no"

        # Add a leading space so tokenization is cleaner for many LMs.
        full = prompt + " " + answer

        tok_full = self.tok(
            full,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        tok_prompt = self.tok(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        input_ids = tok_full["input_ids"]
        attention_mask = tok_full["attention_mask"]

        # Mask prompt part
        prompt_len = len(tok_prompt["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # If truncation caused mismatch, align sizes
        labels = labels[: len(input_ids)]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def collate_fn(batch):
    # Pad to max length in batch
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

@torch.inference_mode()
def predict_yesno(model, tokenizer, examples, batch_size=8, max_new_tokens=3):
    model.eval()
    preds = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        prompts = [build_prompt(ex["text"]) for ex in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # decode suffix
        for i in range(gen.size(0)):
            true_prompt_len = int(inputs["attention_mask"][i].sum().item())
            suffix_ids = gen[i, true_prompt_len:]
            suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True).strip().lower()
            # pick first yes/no occurrence
            if "yes" in suffix and (suffix.find("yes") < suffix.find("no") or "no" not in suffix):
                preds.append("yes")
            elif "no" in suffix:
                preds.append("no")
            else:
                preds.append("no")
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_path", default="trans_eq_data/train.json")
    parser.add_argument("--valid_path", default="trans_eq_data/valid.json")
    parser.add_argument("--test_path", default="trans_eq_data/test.json")

    parser.add_argument("--output_dir", default="qwen_lora_out")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train_bs", type=int, default=2)
    parser.add_argument("--eval_bs", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")

    # LoRA settings
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    train_data = load_json_list(args.train_path)
    valid_data = load_json_list(args.valid_path)
    test_data  = load_json_list(args.test_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.fp16 and torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)

    # --- LoRA via PEFT ---
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise SystemExit(
            "peft is not installed. Install with: pip install peft\n"
            "Also recommended: pip install accelerate\n"
        ) from e

    # Target modules: Qwen-style attention proj layers commonly include q_proj/k_proj/v_proj/o_proj
    # Some variants use different names; PEFT will warn if not found.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = YesNoSFTDataset(train_data, tokenizer, max_length=args.max_length)
    valid_ds = YesNoSFTDataset(valid_data, tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # --- Evaluate on test set with generation accuracy ---
    gold = [ex["label"] for ex in test_data]
    preds = predict_yesno(model, tokenizer, test_data, batch_size=args.eval_bs)
    acc = np.mean([p == g for p, g in zip(preds, gold)])
    print(f"Qwen LoRA-finetuned accuracy on test set: {acc:.4f}")

if __name__ == "__main__":
    main()
