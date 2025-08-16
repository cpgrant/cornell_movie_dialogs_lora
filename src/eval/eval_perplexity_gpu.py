# eval/eval_perplexity_gpu.py
import math, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_model(model_dir, lora_dir=None):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    if lora_dir and not lora_dir.endswith("-merged"):
        model = PeftModel.from_pretrained(model, lora_dir)

    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return tok, model

def collate_instruction(batch, tok, max_len, user_key, out_key):
    texts = []
    prefixes = []
    for ex in batch:
        instr = (ex.get(user_key) or "")
        outp  = (ex.get(out_key)  or "")
        prefix = f"### Instruction:\n{instr}\n\n### Input:\n\n### Response:\n"
        full   = prefix + outp
        prefixes.append(prefix)
        texts.append(full)

    enc_full = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc_pref = tok(prefixes, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    labels = enc_full["input_ids"].clone()
    # mask everything up to the end of the prefix (user/system part)
    # also mask pads
    pad_id = tok.pad_token_id
    for i in range(labels.size(0)):
        pref_len = enc_pref["input_ids"][i].ne(pad_id).sum().item()
        labels[i, :pref_len] = -100
        labels[i, labels[i] == pad_id] = -100

    return {"input_ids": enc_full["input_ids"], "attention_mask": enc_full["attention_mask"], "labels": labels}

def collate_chat(batch, tok, sys_msg, max_len, user_key, out_key):
    # build chat strings and a matching "prefix" that ends right before assistant response
    texts, prefixes = [], []
    for ex in batch:
        user = (ex.get(user_key) or "")
        ans  = (ex.get(out_key)  or "")
        msgs = []
        if sys_msg:
            msgs.append({"role":"system","content":sys_msg})
        msgs.append({"role":"user","content":user})
        # prefix includes roles up to assistant start, *not* the answer
        prefix_text = tok.apply_chat_template(msgs + [{"role":"assistant","content":""}], tokenize=False, add_generation_prompt=False)
        full_text   = tok.apply_chat_template(msgs + [{"role":"assistant","content":ans}], tokenize=False, add_generation_prompt=False)
        prefixes.append(prefix_text)
        texts.append(full_text)

    enc_full = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc_pref = tok(prefixes, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    labels = enc_full["input_ids"].clone()
    pad_id = tok.pad_token_id
    for i in range(labels.size(0)):
        pref_len = enc_pref["input_ids"][i].ne(pad_id).sum().item()
        labels[i, :pref_len] = -100
        labels[i, labels[i] == pad_id] = -100

    return {"input_ids": enc_full["input_ids"], "attention_mask": enc_full["attention_mask"], "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--lora_dir", default=None)
    ap.add_argument("--data", default="data/validation.jsonl")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--format", choices=["chat","instruction"], default="chat")
    ap.add_argument("--system_msg", default=None)
    ap.add_argument("--user_key", default="prompt")
    ap.add_argument("--out_key",  default="response")
    args = ap.parse_args()

    tok, model = load_model(args.model_dir, args.lora_dir)
    ds = load_dataset("json", data_files=args.data, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    if args.format == "chat":
        collate = lambda b: collate_chat(b, tok, args.system_msg, args.max_len, args.user_key, args.out_key)
    else:
        collate = lambda b: collate_instruction(b, tok, args.max_len, args.user_key, args.out_key)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    loss_sum, tok_count = 0.0, 0
    autocast_dtype = torch.bfloat16 if next(model.parameters()).dtype == torch.bfloat16 else None

    pbar = tqdm(dl, desc="Evaluating PPL", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    out = model(**batch)
            else:
                out = model(**batch)

            # count only non-pad tokens for proper aggregation
            num_tokens = (batch["labels"] != tok.pad_token_id).sum().item()
            loss_sum += out.loss.item() * num_tokens
            tok_count += num_tokens
            pbar.set_postfix(tokens=tok_count)

    ppl = math.exp(loss_sum / max(1, tok_count))
    print(f"Perplexity: {ppl:.3f}")

if __name__ == "__main__":
    main()
