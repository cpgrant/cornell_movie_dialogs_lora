import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
BASE="microsoft/phi-3-mini-4k-instruct"; ADAPTER="out-cornell-phi3"
tok=AutoTokenizer.from_pretrained(BASE)
base=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model=PeftModel.from_pretrained(base, ADAPTER); model.eval()
def chat(history):
    user = history[-1][0]
    if hasattr(tok, "apply_chat_template"):
        msgs=[{"role":"user","content":user}]
        prompt=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt=f"You: {user}\nBot:"
    ids=tok(prompt, return_tensors="pt").to(model.device)
    out=model.generate(**ids, max_new_tokens=160, do_sample=True, top_p=0.9, temperature=0.8, repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    history[-1][1]=tok.decode(out[0], skip_special_tokens=True)
    return history
gr.ChatInterface(fn=chat, title="Cornell Movie Dialogs LoRA").launch()
