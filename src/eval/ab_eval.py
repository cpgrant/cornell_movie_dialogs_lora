# eval/ab_eval.py
import argparse, json, os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


# Stopping criteria (path varies by version)
try:
    from transformers.generation import StoppingCriteria, StoppingCriteriaList
except ImportError:
    from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

# Logits processors (name/path varies by version)
# Try BadWordsLogitsProcessor first; fall back to NoBadWordsLogitsProcessor
try:
    from transformers.generation.logits_process import LogitsProcessorList, BadWordsLogitsProcessor as _BadWordsProc
except Exception:
    from transformers.generation.logits_process import LogitsProcessorList, NoBadWordsLogitsProcessor as _BadWordsProc


class StopAfterNNewlines(StoppingCriteria):
    def __init__(self, tok, input_len, n=2):
        self.tok = tok; self.input_len = input_len; self.n = n
    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0, self.input_len:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        # stop once we have N non-empty lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return len(lines) >= self.n
    

def load(model_dir, lora_dir=None):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    )
    if lora_dir and not lora_dir.endswith("-merged"):
        model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return tok, model

def chat(tok, model, prompt, system, args):
    msgs = []
    if system: msgs.append({"role":"system","content":system})
    msgs.append({"role":"user","content":prompt})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt")
    enc = {k:v.to(model.device) for k,v in enc.items()}

    input_len = enc["input_ids"].shape[1]
    stops = StoppingCriteriaList([StopAfterNNewlines(tok, input_len, n=2)])

    banned = ["hell","Hell","HELL","damn","Damn","kill","murder","suicide","sex","SEX"]

    # build processors only if we actually have tokens to ban
    bad_ids = [ids for ids in tok(banned, add_special_tokens=False).input_ids if ids]
    if bad_ids:
        try:
            procs = LogitsProcessorList([_BadWordsProc(bad_ids, eos_token_id=tok.eos_token_id)])
        except TypeError:
            # older transformers versions don't take eos_token_id here
            procs = LogitsProcessorList([_BadWordsProc(bad_ids)])
    else:
        procs = LogitsProcessorList()


    out = model.generate(
        **enc,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        logits_processor=procs,
        stopping_criteria=stops,
    )
    gen = out[0][input_len:]
    # keep exactly two non-empty lines
    text = tok.decode(gen, skip_special_tokens=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:2])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="eval/eval_prompts.jsonl")
    ap.add_argument("--base", default="microsoft/phi-3-mini-4k-instruct")
    ap.add_argument("--finetuned", default="./out-cornell-phi3-merged-36500")
    ap.add_argument("--lora_dir", default=None, help="Optional: use adapter instead of merged")
    ap.add_argument("--system", default="You are friendly and PG. Write exactly two short lines of dialogue for the scenario.")
    ap.add_argument("--out", default="eval/ab_results.jsonl")

    # NEW knobs
    ap.add_argument("--temperature", type=float, default=0.65)
    ap.add_argument("--top_p", type=float, default=0.85)
    ap.add_argument("--max_new_tokens", type=int, default=64)  # set 48 if you want even tighter
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tok_b, model_b = load(args.base, None)
    tok_f, model_f = load(args.finetuned, None if args.finetuned.endswith("-merged") else args.lora_dir)

    wins = {"base":0, "finetuned":0, "tie":0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.prompts) as f, open(args.out, "w") as w:
        for line in f:
            p = json.loads(line)["prompt"]
            rb = chat(tok_b, model_b, p, args.system, args)
            rf = chat(tok_f, model_f, p, args.system, args)

            print("\nPROMPT:", p)
            print("\n[1] BASE:\n", rb)
            print("\n[2] FINETUNED:\n", rf)

            choice = input("\nWinner? (1=base, 2=finetuned, 0=tie/skip): ").strip()
            if choice == "1": wins["base"] += 1
            elif choice == "2": wins["finetuned"] += 1
            else: wins["tie"] += 1

            w.write(json.dumps({"prompt": p, "base": rb, "finetuned": rf, "winner": choice}) + "\n")

    total = sum(wins.values()) or 1
    print("\nWin-rate (finetuned): {:.1f}%".format(100*wins["finetuned"]/total))
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
