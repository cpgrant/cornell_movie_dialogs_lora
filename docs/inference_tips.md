Exactly — you can steer a LoRA-finetuned model a lot at inference. Here are the most impactful knobs (and sane ranges) for dialogue-style models like your Phi-3 + Cornell adapter:

### Core sampling knobs

* **`temperature`** (0.2–1.2): scales randomness. Lower = focused/boring; higher = creative/erratic.
* **`top_p`** (0.8–0.95): nucleus sampling; keep the smallest token set whose probs sum to *p*. Lower = safer.
* **`top_k`** (20–200): keep only the top-k probable tokens. Use with or instead of `top_p`.
* **`do_sample`**: `True` enables the above sampling; `False` = deterministic (greedy/beam).
* **`max_new_tokens`**: hard cap on generation length.

### Repetition & structure

* **`repetition_penalty`** (1.05–1.25): discourages verbatim loops. Too high can degrade quality.
* **`no_repeat_ngram_size`** (2–4): disallows repeating n-grams; strong but sometimes over-restrictive.
* **`length_penalty`** (beams only): >1 favors longer text, <1 favors shorter.
* **Stop sequences**: cut generation when you see strings like `"\nYou:"`, `"User:"`, etc.

### Deterministic alternatives

* **Greedy**: `do_sample=False` (stable but dull).
* **Beam search**: `num_beams=3–6`, optionally `early_stopping=True`. Better structure, less spontaneity.

### Less common but useful

* **`typical_p`** (0.8–0.95): keeps tokens of “typical” probability mass; alternative to top-p.
* **`penalty_alpha`** (contrastive search): try 0.6–0.8 with `top_k=4–8` (no sampling); concise and coherent.
* **Classifier-free guidance** (`guidance_scale`) exists in some implementations; not standard in HF text-gen.

### Practical recipes

**Tight + safe (FAQ-ish):**

```python
out = model.generate(**ids, do_sample=False, max_new_tokens=120)
```

**Conversational but controlled:**

```python
out = model.generate(**ids, do_sample=True, temperature=0.7, top_p=0.9,
                     max_new_tokens=140, repetition_penalty=1.1, no_repeat_ngram_size=3)
```

**Creative dialogue (screenplay vibe):**

```python
out = model.generate(**ids, do_sample=True, temperature=0.9, top_p=0.92, top_k=100,
                     max_new_tokens=160, repetition_penalty=1.07)
```

**Concise & coherent (contrastive):**

```python
out = model.generate(**ids, do_sample=False, penalty_alpha=0.6, top_k=6,
                     max_new_tokens=120)
```

### Tips for your script

* Keep using the **chat template** (`apply_chat_template`)—it usually improves turn-taking.
* Add **stop strings** that match your fallback prefix:

  ```python
  stop = ["\nYou:", "\nUser:", "\nBot:"]
  out = model.generate(**ids, eos_token_id=tok.eos_token_id, max_new_tokens=140)
  # After decode: truncate at first occurrence of any stop string.
  ```
* Set a **seed** during testing for comparability:

  ```python
  torch.manual_seed(42)
  ```
* If you merge LoRA (`-merged`), you can see slightly faster inference and simpler deployment; behavior is otherwise the same API-wise.

If you tell me the exact feel you want (e.g., “snappy one-liners,” “dramatic back-and-forth,” “PG, no slang”), I’ll give you a tuned parameter set + prompt pattern to match it.
