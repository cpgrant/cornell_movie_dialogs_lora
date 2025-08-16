# save as scripts/eval_perplexity.py
import math, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./out-cornell-phi3-merged"  # or base model to compare
DATA = "data/validation.jsonl"           # your file

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")

ds = load_dataset("json", data_files=DATA, split="train")

def make_text(x):
    # match your training template (adjust as needed)
    instr = x.get("instruction","")
    inp   = x.get("input","")
    outp  = x.get("output","")
    return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{outp}"

loss_sum, tok_count = 0.0, 0
for ex in ds:
    text = make_text(ex)
    enc = tok(text, return_tensors="pt")
    enc = {k:v.to(model.device) for k,v in enc.items()}
    with model.disable_adapter() if hasattr(model,"disable_adapter") else contextlib.nullcontext():
        # nothing special; just compute LM loss on its own continuation
        labels = enc["input_ids"].clone()
        out = model(**enc, labels=labels)
    loss = out.loss.item()
    loss_sum += loss * labels.numel()
    tok_count += labels.numel()

ppl = math.exp(loss_sum / tok_count)
print("Perplexity:", ppl)
