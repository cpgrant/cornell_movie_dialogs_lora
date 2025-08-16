import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

print("Gradio version:", gr.__version__)


BASE = "microsoft/phi-3-mini-4k-instruct"
ADAPTER = "out-cornell-phi3"

tok = AutoTokenizer.from_pretrained(BASE)
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

if ADAPTER.endswith("-merged"):
    model = AutoModelForCausalLM.from_pretrained(ADAPTER, torch_dtype=dtype, device_map="auto")
else:
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(base, ADAPTER)

model.eval()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def chat(messages):  # messages = list of {"role": "...", "content": "..."}
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = ""
        for m in messages:
            if m["role"] == "user":
                prompt += f"You: {m['content']}\n"
            elif m["role"] == "assistant":
                prompt += f"Bot: {m['content']}\n"
        prompt += "Bot:"

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
        )

    reply = tok.decode(out[0], skip_special_tokens=True)
    return messages + [{"role": "assistant", "content": reply}]

gr.ChatInterface(
    fn=chat,
    title="Cornell Movie Dialogs LoRA",
    type="messages",
).launch()
