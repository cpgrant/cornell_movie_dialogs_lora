# inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, sys, argparse
from typing import List, Dict

DEFAULT_BASE = "microsoft/phi-3-mini-4k-instruct"

def load_model(model_dir, lora_dir=None, dtype="bf16"):
    # dtype selection
    torch_dtype = None
    if dtype == "bf16" and torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    elif dtype == "fp16" and torch.cuda.is_available():
        torch_dtype = torch.float16

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    if lora_dir and not lora_dir.endswith("-merged"):
        model = PeftModel.from_pretrained(model, lora_dir)

    model.eval()
    # small speed boosts (e.g., 4090)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return tok, model

def build_prompt(tok, messages: List[Dict], max_input_tokens: int):
    """
    Apply chat template and truncate from the left if the prompt is too long.
    """
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][0]
    if input_ids.shape[0] > max_input_tokens:
        # left-truncate tokens to fit the model context window
        input_ids = input_ids[-max_input_tokens:]
    return input_ids.unsqueeze(0)

def generate(tok, model, input_ids, max_new_tokens, temperature, top_p,
             no_repeat_ngram_size, repetition_penalty, length_penalty):
    input_ids = input_ids.to(model.device)
    in_len = input_ids.shape[1]
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    gen = out[0][in_len:]
    return tok.decode(gen, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=DEFAULT_BASE, help="Base or merged model dir")
    ap.add_argument("--lora_dir", default=None, help="LoRA adapter dir (omit if merged)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--length_penalty", type=float, default=1.15)
    ap.add_argument("--max_input_tokens", type=int, default=2048, help="Budget for the prompt/history tokens")
    ap.add_argument("--system_msg", default=(
        "You are a friendly, witty movie-style conversationalist. "
        "Stay coherent across turns, avoid insults, and answer in 3–5 sentences unless asked otherwise."
    ))
    args = ap.parse_args()

    tok, model = load_model(args.model_dir, args.lora_dir, args.dtype)

    # conversation history (system + alternating user/assistant)
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

            input_ids = build_prompt(tok, history, args.max_input_tokens)
            resp = generate(
                tok, model, input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
            )

            print(f"Model: {resp}\n")
            history.append({"role": "assistant", "content": resp})

        except KeyboardInterrupt:
            print()
            sys.exit(0)

if __name__ == "__main__":
    main()
