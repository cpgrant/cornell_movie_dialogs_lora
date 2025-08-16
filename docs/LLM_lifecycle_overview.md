---

# 🌐 LLM Lifecycle Overview

```
        ┌───────────────────┐
        │   1. Dataset      │
        │  (collect & prep) │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   2. Pretraining  │
        │ (general language │
        │   knowledge)      │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   3. Fine-tuning  │
        │  (adapt to tasks, │
        │   style, domain)  │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   4. Inference    │
        │ (generation,      │
        │ decoding params,  │
        │   prompts)        │
        └───────────────────┘
```

---

# 📊 Phase by Phase

### **1. Dataset**

* **Goal:** Gather text data
* **Examples:** web text, Wikipedia, books, movie dialogues, Q\&A pairs
* **Your project:** Cornell Movie Dialogs → `train.jsonl`, `val.jsonl`, `test.jsonl`

---

### **2. Pretraining**

* **Goal:** Teach base language & world knowledge
* **What you control:** Not usually (done by labs)
* **Output:** Base model (`microsoft/phi-3-mini-4k-instruct`)

---

### **3. Fine-Tuning**

* **Goal:** Specialize to task/style
* **Methods:** Full finetune, LoRA/QLoRA, Instruction Tuning, RLHF
* **Your setup:** LoRA on Cornell → `out-cornell-phi3` + merged `out-cornell-phi3-merged`

---

### **4. Inference**

* **Goal:** Generate text in your app
* **Knobs:**

  * `temperature`, `top_p`, `top_k` → creativity vs. focus
  * `max_new_tokens`, stopping criteria → length control
  * `repetition_penalty`, `no_repeat_ngram_size` → avoid loops
  * System message, chat template, history length → tone & coherence
* **Your case:** interactive `inference.py` with conversation memory

---

👉 The way to read this:

* **Dataset + Pretraining** = what the model *knows*
* **Fine-tuning** = how the model *behaves*
* **Inference** = how you *let it speak*

---

Would you like me to make this into a **polished diagram image** (boxes, arrows, color-coded) so you can drop it into your notes or slides?
