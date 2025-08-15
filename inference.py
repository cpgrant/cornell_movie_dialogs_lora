from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, sys, os

BASE = "microsoft/phi-3-mini-4k-instruct"
ADAPTER = "out-cornell-phi3"           # <- fixed folder name
# If you merged adapters into the base weights, set:
# ADAPTER = "out-cornell-phi3-merged"

def load_model():
    # quick sanity: local path exists?
    if not (ADAPTER.endswith("-merged") or os.path.exists(ADAPTER)):
        raise FileNotFoundError(f"Adapter path not found: {ADAPTER}")

    tok = AutoTokenizer.from_pretrained(BASE)
    # prefer BF16 on 4090
    dtype = torch.bfloat16

    if ADAPTER.endswith("-merged"):
        # load merged model directly
        model = AutoModelForCausalLM.from_pretrained(ADAPTER, torch_dtype=dtype, device_map="auto")
    else:
        # load base, then apply LoRA adapter from local folder
        model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype, device_map="auto")
        model = PeftModel.from_pretrained(model, ADAPTER)

    model.eval()
    # small speed boost on 4090
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return tok, model

def format_chat(tok, user_text):
    # use the model’s chat template if available
    if hasattr(tok, "apply_chat_template"):
        msgs = [{"role": "user", "content": user_text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # fallback formatting
    return f"You: {user_text}\nBot:"

def generate(tok, model, prompt):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    tok, model = load_model()
    print("Type your message; Ctrl+C to quit.")
    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            prompt = format_chat(tok, user)
            print(generate(tok, model, prompt))
        except KeyboardInterrupt:
            sys.exit(0)
