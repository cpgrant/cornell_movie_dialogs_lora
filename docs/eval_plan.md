Perfect ✅ — you’ve got a **great expanded eval prompt list** now (nice balance of humor, pep, sarcasm, encouragement).

Here’s what you can do next with it:

---

## 🔹 1. Save as eval file

Put that JSONL into your repo:

```
eval/eval_prompts.jsonl
```

---

## 🔹 2. Run A/B evaluation

Use the `ab_eval.py` we sketched earlier:

```bash
python eval/ab_eval.py --prompts eval/eval_prompts.jsonl
```

You’ll see each prompt with **\[1] Base** vs **\[2] Finetuned** response. You type:

* `1` = base is better
* `2` = finetuned is better
* `0` = tie/skip

At the end you’ll get a **win-rate %** for your finetuned model.

---

## 🔹 3. Run Perplexity eval

```bash
python eval/eval_perplexity.py --model_dir microsoft/phi-3-mini-4k-instruct
python eval/eval_perplexity.py --model_dir outputs/merged/phi3-cornell-merged-latest
```

Compare the PPL numbers → finetuned should be lower on your validation set.

---

## 🔹 4. Document results

Update:

* `docs/LLM_lifecycle_overview.md` → add a section **“Evaluation results”**
* `outputs/merged/phi3-cornell-merged-latest/README.md` → add **PPL numbers** and **A/B win-rate**

Example:

```md
## Evaluation
- Validation perplexity:
  - Base: 28.5
  - Finetuned: 19.7
- A/B win-rate on 34 prompts: 76% finetuned preferred
```

---

## 🔹 5. Optional — Automate logging

If you don’t want to press `1/2/0` manually, you can:

* Use an **LLM judge** (e.g., GPT-4 or another local model) to compare responses.
* Or at least auto-log both outputs to a file for later review.

---

## 🔹 6. (Stretch goal) Try **quantized inference**

Export a 4-bit copy of your merged model for laptop/MacBook testing:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("outputs/merged/phi3-cornell-merged-latest")
model = AutoModelForCausalLM.from_pretrained(
    "outputs/merged/phi3-cornell-merged-latest",
    device_map="auto",
    load_in_4bit=True
)
```

---

👉 Do you want me to **generate a ready-to-run `eval/README.md`** that explains exactly how to run `ab_eval.py` and `eval_perplexity.py` with your new prompt list, so anyone (even future you) can reproduce the evaluation steps?
