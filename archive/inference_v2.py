from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, sys, os, argparse

DEFAULT_BASE = "microsoft/phi-3-mini-4k-instruct"

def load_model(model_dir, lora_dir=None, dtype="bf16"):
    torch_dtype = torch.bfloat16 if dtype == "bf16" and torch.cuda.is_available() else None

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    if lora_dir and not lora_dir.endswith("-merged"):
        model = PeftModel.from_pretrained(model, lora_dir)

    model.eval()
    # small speed boosts (4090)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return tok, model

def build_prompt(tok, user_text, system_msg=None):
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_text})

    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback if no chat template is present
    sys_prefix = f"[System: {system_msg}]\n" if system_msg else ""
    return f"{sys_prefix}User: {user_text}\nAssistant:"

def generate(tok, model, prompt, max_new_tokens, temperature, top_p,
             no_repeat_ngram_size, repetition_penalty, length_penalty):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    in_len = ids["input_ids"].shape[1]
    out = model.generate(
        **ids,
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
    # strip the prompt to avoid echo
    gen = out[0][in_len:]
    return tok.decode(gen, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=DEFAULT_BASE, help="Base or merged model dir")
    ap.add_argument("--lora_dir", default=None, help="LoRA adapter dir (omit if merged)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.07)
    ap.add_argument("--length_penalty", type=float, default=1.10)
    ap.add_argument("--system_msg", default=(
        "You are a witty conversationalist. Answer in 3–4 complete sentences unless asked otherwise. "
        "Keep continuity and avoid repeating the user's words verbatim."
    ))
    args = ap.parse_args()

    # If user passed the merged folder as --model_dir, keep --lora_dir None
    tok, model = load_model(args.model_dir, args.lora_dir, args.dtype)

    print("Type your message; Ctrl+C to quit.")
    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            prompt = build_prompt(tok, user, system_msg=args.system_msg)
            resp = generate(
                tok, model, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
            )
            print(f"Model: {resp}\n")
        except KeyboardInterrupt:
            print()
            sys.exit(0)

if __name__ == "__main__":
    main()
