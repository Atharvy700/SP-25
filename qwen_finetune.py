import argparse
import json
import os
import random
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, TaskType


# ----------------------------
# Utils: load json or jsonl
# ----------------------------
def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {path}")


# ----------------------------
# Build prompts for evaluation
# ----------------------------
SYSTEM = "You are a binary classifier. Answer ONLY “yes” or “no”."


def make_eval_prompt(text: str) -> str:
    return f"Text: {text}\nLabel (yes/no):"


def normalize_yesno(s: str) -> str:
    s = s.strip().lower()
    # take first token-ish
    s = s.split()[0] if s else ""
    # common punctuation
    s = s.strip(".,;:!\"'()[]{}")
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    return ""


@torch.no_grad()
def eval_accuracy(model, tokenizer, test_items, max_new_tokens=3, temperature=0.0, device="cuda", max_test=None):
    model.eval()
    n = 0
    correct = 0

    if max_test is not None:
        test_items = test_items[:max_test]

    for ex in test_items:
        gold = ex["label"].strip().lower()

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": make_eval_prompt(ex["text"])},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

        out = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = normalize_yesno(out)

        if pred == gold:
            correct += 1
        n += 1

        if n % 200 == 0:
            print(f"[eval] {n} done | acc so far = {correct/n:.4f}")

    acc = correct / max(n, 1)
    return acc


# ----------------------------
# Tokenization for SFT chat jsonl
# Your train_ft.jsonl has: {"messages":[...]}
# We train only on assistant tokens by masking labels.
# ----------------------------
def tokenize_chat_example(example, tokenizer, max_length: int):
    messages = example["messages"]
    # Build full chat text
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    input_ids = enc["input_ids"]

    # Build labels but mask everything up to (and including) the last user message.
    # Simplest robust approach: find the final assistant content string and only supervise that span.
    # We'll locate the assistant content in the rendered text and mask earlier chars by retokenizing prefix.
    # (Works well enough for this binary "yes/no" setup.)
    last_assistant = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_assistant = m.get("content", "")
            break

    labels = [-100] * len(input_ids)

    if last_assistant is not None and len(last_assistant.strip()) > 0:
        # Try to find where assistant answer starts by rendering up to assistant message start
        prefix_msgs = []
        seen_assistant = False
        for m in messages:
            if m.get("role") == "assistant" and not seen_assistant:
                seen_assistant = True
                break
            prefix_msgs.append(m)

        prefix_text = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True)
        prefix_ids = tokenizer(prefix_text, truncation=True, max_length=max_length, padding=False)["input_ids"]

        # The generation prompt ends right before assistant answer; supervise everything after that.
        start = min(len(prefix_ids), len(input_ids))
        for i in range(start, len(input_ids)):
            labels[i] = input_ids[i]

    return {"input_ids": input_ids, "attention_mask": enc["attention_mask"], "labels": labels}


class PadCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, features):
        # pad input_ids/attention_mask/labels to max length in batch
        maxlen = max(len(f["input_ids"]) for f in features)

        def pad(seq, pad_id):
            return seq + [pad_id] * (maxlen - len(seq))

        input_ids = [pad(f["input_ids"], self.tok.pad_token_id) for f in features]
        attention_mask = [pad(f["attention_mask"], 0) for f in features]
        labels = [pad(f["labels"], -100) for f in features]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train_path", default="trans_eq_data/train_ft.jsonl")
    ap.add_argument("--valid_path", default="trans_eq_data/valid_ft.jsonl")
    ap.add_argument("--test_path", default="trans_eq_data/test.json")
    ap.add_argument("--output_dir", default="qwen_lora_run")

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--train_bs", type=int, default=2)
    ap.add_argument("--eval_bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--max_valid", type=int, default=None)
    ap.add_argument("--max_test", type=int, default=None)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # IMPORTANT: do NOT use 8bit/4bit. Avoid bitsandbytes completely.
    dtype = torch.float16 if args.fp16 else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    # LoRA config (Qwen-style proj names)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Load datasets
    train_items = load_json_or_jsonl(args.train_path)
    valid_items = load_json_or_jsonl(args.valid_path)
    test_items = load_json_or_jsonl(args.test_path)

    if args.max_train is not None:
        train_items = train_items[:args.max_train]
    if args.max_valid is not None:
        valid_items = valid_items[:args.max_valid]

    train_ds = Dataset.from_list(train_items).map(lambda ex: tokenize_chat_example(ex, tokenizer, args.max_length))
    valid_ds = Dataset.from_list(valid_items).map(lambda ex: tokenize_chat_example(ex, tokenizer, args.max_length))

    collator = PadCollator(tokenizer)

    # Training args: use eval_strategy (matches your BERT script style)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        fp16=args.fp16 and device == "cuda",
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate accuracy on test.json (yes/no)
    # Load the trained adapter model already in memory
    acc = eval_accuracy(
        model=model,
        tokenizer=tokenizer,
        test_items=test_items,
        max_new_tokens=3,
        temperature=0.0,
        device=device,
        max_test=args.max_test,
    )
    print(f"\nTEST ACCURACY = {acc:.6f}")


if __name__ == "__main__":
    main()
