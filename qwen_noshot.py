#!/usr/bin/env python3
# qwen_noshot.py
import argparse
import json
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

YES_SET = {"yes", "y", "true", "1"}
NO_SET  = {"no", "n", "false", "0"}

def load_json_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(example_text: str) -> str:
    # Keep it simple + force constrained answer.
    return (
        "You are a careful reasoning assistant.\n"
        "Task: Answer the final question with exactly one word: yes or no.\n\n"
        f"{example_text}\n\n"
        "Answer (yes/no):"
    )

def extract_yes_no(generated: str):
    """
    Extract the first yes/no-like token from model output.
    Returns 'yes'/'no' or None.
    """
    s = generated.strip().lower()

    # Common patterns: "yes", "yes.", "Answer: yes", "yes\n"
    m = re.search(r"\b(yes|no)\b", s)
    if not m:
        return None
    return m.group(1)

@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens=3, temperature=0.0):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    # Deterministic by default
    do_sample = temperature > 0
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Only decode the newly generated suffix for each item
    out_texts = []
    for i in range(gen.size(0)):
        prompt_len = inputs["input_ids"][i].size(0)
        # If padded, prompt_len includes padding; safer: find non-pad length
        attn = inputs["attention_mask"][i]
        true_prompt_len = int(attn.sum().item())
        suffix_ids = gen[i, true_prompt_len:]
        out_texts.append(tokenizer.decode(suffix_ids, skip_special_tokens=True))
    return out_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HF model id (causal LM). Use an Instruct model if possible.")
    parser.add_argument("--test_path", default="trans_eq_data/test.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (optional)")
    args = parser.parse_args()

    test_data = load_json_list(args.test_path)

    # Device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device in {"cuda", "mps"} else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    gold = [ex["label"] for ex in test_data]
    preds = []

    prompts = [build_prompt(ex["text"]) for ex in test_data]

    # Batched generation
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start:start + args.batch_size]
        suffixes = generate_batch(
            model, tokenizer, batch_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        for suf in suffixes:
            yn = extract_yes_no(suf)
            preds.append(yn if yn in {"yes", "no"} else "no")  # fallback

    acc = np.mean([p == g for p, g in zip(preds, gold)])
    print(f"Qwen zero-shot accuracy on test set: {acc:.4f}")

    # Optional: show a few examples
    for i in range(3):
        print("\n---")
        print("GOLD:", gold[i])
        print("PRED:", preds[i])
        print("GEN :", preds[i])

if __name__ == "__main__":
    main()
