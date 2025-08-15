from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, glob

BASE="microsoft/phi-3-mini-4k-instruct"
ADAPTER="out-cornell-phi3"   # or out-cornell-phi3-merged

tok=AutoTokenizer.from_pretrained(BASE)
if ADAPTER.endswith("-merged"):
    model=AutoModelForCausalLM.from_pretrained(ADAPTER, torch_dtype=torch.bfloat16, device_map="auto")
else:
    base=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
    model=PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def gen(s):
    if hasattr(tok, "apply_chat_template"):
        msgs=[{"role":"user","content":s}]
        text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text=f"You: {s}\nBot:"
    ids=tok(text, return_tensors="pt").to(model.device)
    out=model.generate(**ids, max_new_tokens=140, do_sample=True, top_p=0.9, temperature=0.8, repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    print("You:", s)
    print("Bot:", tok.decode(out[0], skip_special_tokens=True), "\n")

tests=[
 "I can’t believe you just did that! Do you know what this means?",
 "The car won’t start and we’re in the middle of nowhere. What do we do?",
 "Don’t walk out that door—not after everything we’ve been through.",
 "You’re my grumpy older brother on a desert road trip. We just blew a tire."
]
for t in tests: gen(t)
