# inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, sys, argparse, random
from typing import List, Dict, Optional

DEFAULT_BASE = "microsoft/phi-3-mini-4k-instruct"

def _pick_dtype(dtype: str) -> Optional[torch.dtype]:
    if torch.cuda.is_available():
        if dtype == "bf16":
            return torch.bfloat16
        if dtype == "fp16":
            return torch.float16
    # fall back to fp32 on CPU or if bf16/fp16 not available
    return None

def load_model(model_dir: str, lora_dir: Optional[str], dtype: str = "bf16"):
    torch_dtype = _pick_dtype(dtype)

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    # ensure pad token exists
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # Load base or merged model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, device_map="auto"
    )

    # Attach LoRA adapter only if provided and not already merged
    if lora_dir and not lora_dir.endswith("-merged"):
        model = PeftModel.from_pretrained(model, lora_dir)

    model.eval()
    # speed hints
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return tok, model

def build_input_ids(tok, messages: List[Dict], max_input_tokens: int, use_template: bool = True):
    if use_template and hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # simple fallback formatting
        lines = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            prefix = {"system":"[SYSTEM]", "user":"You", "assistant":"Model"}.get(role, role.capitalize())
            lines.append(f"{prefix}: {content}")
        text = "\n".join(lines) + "\nModel:"
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][0]
    if input_ids.shape[0] > max_input_tokens:
        input_ids = input_ids[-max_input_tokens:]  # left truncate
    return input_ids.unsqueeze(0)

def postprocess(text: str, stops: List[str]) -> str:
    if not stops:
        return text
    cut = len(text)
    for s in stops:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].rstrip()

def generate(tok, model, input_ids, max_new_tokens, temperature, top_p,
             no_repeat_ngram_size, repetition_penalty, stops: List[str]):
    # move to the same device as model
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0][input_ids.shape[1]:]
    raw = tok.decode(gen, skip_special_tokens=True)
    return postprocess(raw, stops)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=DEFAULT_BASE, help="Base or merged model dir")
    ap.add_argument("--lora_dir", default=None, help="LoRA adapter dir (omit if merged)")
    ap.add_argument("--adapter", default=None, help="Alias for --lora_dir")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--seed", type=int, default=42)

    # generation defaults (coherent dialogue)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)

    ap.add_argument("--max_input_tokens", type=int, default=2048)
    ap.add_argument("--no_template", action="store_true", help="Disable tokenizer chat template")
    ap.add_argument("--stop", action="append", default=[], help="Add a stop string (repeatable)")
    ap.add_argument("--system_msg", default=(
        "You are witty and concise. Reply in 2–3 sentences like movie dialogue. "
        "Stay coherent across turns and avoid insults."
    ))
    args = ap.parse_args()

    # resolve adapter alias
    if args.adapter and not args.lora_dir:
        args.lora_dir = args.adapter

    # sensible default stops for fallback formatting
    if not args.stop:
        args.stop = ["\nYou:", "\nUser:", "\nModel:", "\nAssistant:"]

    set_seed(args.seed)

    print("Loading model…")
    tok, model = load_model(args.model_dir, args.lora_dir, args.dtype)

    print("--- Inference config ---")
    print(f"model_dir={args.model_dir}")
    print(f"lora_dir={args.lora_dir}")
    print(f"dtype={args.dtype}  seed={args.seed}")
    print(f"temperature={args.temperature}  top_p={args.top_p}  max_new_tokens={args.max_new_tokens}")
    print("------------------------\n")

    history: List[Dict] = []
    if args.system_msg:
        history.append({"role": "system", "content": args.system_msg})

    print("Type your message; Ctrl+C to quit.")
    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            history.append({"role": "user", "content": user})

            input_ids = build_input_ids(
                tok, history, args.max_input_tokens, use_template=not args.no_template
            )
            resp = generate(
                tok, model, input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                stops=args.stop,
            )
            print(f"Model: {resp}\n")
            history.append({"role": "assistant", "content": resp})
        except KeyboardInterrupt:
            print()
            sys.exit(0)

if __name__ == "__main__":
    main()
