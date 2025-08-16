# inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, sys, argparse
from typing import List, Dict

DEFAULT_BASE = "microsoft/phi-3-mini-4k-instruct"

def load_model(model_dir, lora_dir=None, dtype="bf16"):
    torch_dtype = None
    if dtype == "bf16" and torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    elif dtype == "fp16" and torch.cuda.is_available():
        torch_dtype = torch.float16

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    # ensure we have a pad token (some Phi variants don’t)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, device_map="auto"
    )
    if lora_dir and not lora_dir.endswith("-merged"):
        model = PeftModel.from_pretrained(model, lora_dir)

    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return tok, model

def build_input_ids(tok, messages: List[Dict], max_input_tokens: int):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][0]
    if input_ids.shape[0] > max_input_tokens:
        input_ids = input_ids[-max_input_tokens:]  # left-truncate history
    return input_ids.unsqueeze(0)

def generate(tok, model, input_ids, max_new_tokens, temperature, top_p,
             no_repeat_ngram_size, repetition_penalty):
    # Build attention_mask explicitly to avoid warnings & ensure correct masking.
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
    return tok.decode(gen, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=DEFAULT_BASE, help="Base or merged model dir")
    ap.add_argument("--lora_dir", default=None, help="LoRA adapter dir (omit if merged)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16","fp32"])
    # Tighter, more coherent defaults:
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--max_input_tokens", type=int, default=2048)
    ap.add_argument("--system_msg", default=(
        "You are witty and concise. Reply in 2–3 sentences like movie dialogue. "
        "Stay coherent across turns and avoid insults."
    ))
    args = ap.parse_args()

    tok, model = load_model(args.model_dir, args.lora_dir, args.dtype)

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

            input_ids = build_input_ids(tok, history, args.max_input_tokens).to(model.device)
            resp = generate(
                tok, model, input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
            )
            print(f"Model: {resp}\n")
            history.append({"role": "assistant", "content": resp})
        except KeyboardInterrupt:
            print()
            sys.exit(0)

if __name__ == "__main__":
    main()
