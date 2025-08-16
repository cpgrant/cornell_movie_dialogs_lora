Nice structure! You’re basically ready to “productize.” Here’s a tight checklist + drop-in files so you can evaluate, compare, and document cleanly.

# What I’d add next

1. **`eval/` folder** with prompts + two scripts:

* `eval_prompts.jsonl` (10–20 quick dialogue probes)
* `ab_eval.py` (run base vs finetuned, save side-by-side, compute win-rate)
* `eval_perplexity.py` (quick PPL on validation)

2. **Promote `inference_v4.py`** as the default (history, mask, system msg).
3. **README updates**: “Train → Merge → Inference → Eval” one-liners.
4. **Model card** (markdown) for the merged model dir.
5. **Optional**: export a 4-bit quant for laptop deploy.

---

## 1) Create `eval/eval_prompts.jsonl`

Paste this as `eval/eval_prompts.jsonl`:

```json
{"prompt":"Hey, how are you?"}
{"prompt":"Give me a witty comeback to: \"Nice try.\""}
{"prompt":"Two friends are late to class. What do they say to each other?"}
{"prompt":"Your crush waves. Play it cool in one line."}
{"prompt":"Make a sarcastic reply to: \"I love group projects.\""}
{"prompt":"Keep continuity: I'm freezing. — Then what do you say?"}
{"prompt":"Someone says: \"This plan is flawless.\" Respond with playful doubt."}
{"prompt":"Short pep talk before a big game."}
{"prompt":"Decline an invite politely but with humor."}
{"prompt":"Reassure a friend after a bad audition in 2–3 sentences."}
```

## 2) `eval/ab_eval.py`

Runs both models and logs side-by-side; prints a quick win-rate you enter interactively (1 = base wins, 2 = finetuned wins, 0 = tie/skip).

```python
# eval/ab_eval.py
import argparse, json, os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def load(model_dir, lora_dir=None):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
    if lora_dir and not lora_dir.endswith("-merged"):
        model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    return tok, model

def chat(tok, model, prompt, system=None, max_new_tokens=128, temperature=0.7, top_p=0.9):
    msgs = []
    if system:
        msgs.append({"role":"system","content":system})
    msgs.append({"role":"user","content":prompt})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt")
    enc = {k:v.to(model.device) for k,v in enc.items()}
    out = model.generate(
        **enc,
        do_sample=True, temperature=temperature, top_p=top_p,
        no_repeat_ngram_size=3, repetition_penalty=1.15,
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )
    gen = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="eval/eval_prompts.jsonl")
    ap.add_argument("--base", default="microsoft/phi-3-mini-4k-instruct")
    ap.add_argument("--finetuned", default="outputs/merged/phi3-cornell-merged-latest")
    ap.add_argument("--lora_dir", default=None)
    ap.add_argument("--system", default="You are witty and concise. Reply in 2–3 sentences, keep it PG and on-topic.")
    ap.add_argument("--out", default="eval/ab_results.jsonl")
    args = ap.parse_args()

    tok_b, model_b = load(args.base, None)
    tok_f, model_f = load(args.finetuned, None if args.finetuned.endswith("-merged") else args.lora_dir)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wins = {"base":0, "finetuned":0, "tie":0}

    with open(args.prompts) as f, open(args.out,"w") as w:
        for line in f:
            p = json.loads(line)["prompt"]
            rb = chat(tok_b, model_b, p, system=args.system)
            rf = chat(tok_f, model_f, p, system=args.system)
            rec = {"prompt": p, "base": rb, "finetuned": rf}
            print("\nPROMPT:", p)
            print("\n[1] BASE:\n", rb)
            print("\n[2] FINETUNED:\n", rf)
            choice = input("\nWinner? (1=base, 2=finetuned, 0=tie/skip): ").strip()
            if choice == "1": wins["base"] += 1
            elif choice == "2": wins["finetuned"] += 1
            else: wins["tie"] += 1
            w.write(json.dumps({**rec, "winner": choice})+"\n")
    total = sum(wins.values()) or 1
    print("\nWin-rate (finetuned): {:.1f}%".format(100*wins["finetuned"]/total))
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
```

Run:

```bash
python eval/ab_eval.py
```

## 3) `eval/eval_perplexity.py`

Quick PPL on your validation set (compare base vs finetuned).

```python
# eval/eval_perplexity.py
import math, json, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
    model.eval()
    return tok, model

def make_text(ex):
    instr = ex.get("instruction","")
    inp   = ex.get("input","")
    outp  = ex.get("output","")
    return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{outp}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data", default="data/validation.jsonl")
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    tok, model = load(args.model_dir)
    ds = load_dataset("json", data_files=args.data, split="train")

    loss_sum, tok_count = 0.0, 0
    for ex in ds:
        text = make_text(ex)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len)
        enc = {k:v.to(model.device) for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        loss = out.loss.item()
        loss_sum += loss * enc["input_ids"].numel()
        tok_count += enc["input_ids"].numel()
    ppl = math.exp(loss_sum / tok_count)
    print("Perplexity:", round(ppl, 3))

if __name__ == "__main__":
    main()
```

Run (both):

```bash
python eval/eval_perplexity.py --model_dir microsoft/phi-3-mini-4k-instruct
python eval/eval_perplexity.py --model_dir outputs/merged/phi3-cornell-merged-latest
```

---

## 4) README additions (paste into your README.md)

````md
## Quickstart

### Inference (merged)
```bash
source .venv/bin/activate
python inference_v4.py --model_dir outputs/merged/phi3-cornell-merged-latest
````

### Inference (base + LoRA)

```bash
python inference_v4.py --model_dir microsoft/phi-3-mini-4k-instruct --lora_dir ./outputs/adapters/phi3-cornell-lora
```

### A/B Eval

```bash
python eval/ab_eval.py
```

### Perplexity

```bash
python eval/eval_perplexity.py --model_dir outputs/merged/phi3-cornell-merged-latest
```

````

---

## 5) Model card (save as `outputs/merged/phi3-cornell-merged-latest/README.md`)
```md
# Phi-3 Mini (Cornell Dialogs) — LoRA Merged

**Base:** microsoft/phi-3-mini-4k-instruct  
**Adaptation:** LoRA on Cornell Movie Dialogs (style specialization)

## Intended Use
Short, witty dialogue in a movie-bantery tone. Not for factual QA.

## Training
- Dataset: Cornell Movie Dialogs (train/val/test splits)
- Method: LoRA; merged into base for easy inference
- Hardware: 4090 (bf16)

## Inference Tips
- Short quips: `temperature=0.6, top_p=0.85, max_new_tokens=96`
- Longer replies: `temperature=0.7, top_p=0.9, max_new_tokens=192`
- Use chat template + system prompt to steer tone.

## Limitations
- May be sarcastic / snarky
- Not multi-turn coherent without history
- Not suitable for factual tasks or advice

## Evaluation
Include your PPL numbers and A/B win-rate here.
````

---

## 6) (Optional) 4-bit export for laptops

If supported for Phi-3 in your stack:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import quantize  # or use bitsandbytes 4-bit loading at inference

# Or simply load with 4-bit at inference time:
model = AutoModelForCausalLM.from_pretrained(
    "outputs/merged/phi3-cornell-merged-latest",
    device_map="auto",
    load_in_4bit=True
)
```

(If you hit issues with 4-bit + Phi-3, stick to bf16/fp16 on 4090 and MPS on Mac.)

---

If you want, I can also generate a **compact docs/LLM\_lifecycle\_overview\.md** content block that matches your diagram and links to `eval/` and `inference_v4.py`.
