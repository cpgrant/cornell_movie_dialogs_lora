
---

## README.md (updated & polished)

````markdown
# 🎬 Cornell Movie Dialogs – LoRA Fine-Tune (Phi-3 Mini)

Fine-tune **[microsoft/phi-3-mini-4k-instruct](https://huggingface.co/microsoft/phi-3-mini-4k-instruct)** on the **Cornell Movie-Dialogs Corpus** using **LoRA** to get cinematic, multi-turn dialogue.

<p align="left">
  <a href="https://github.com/cpgrant/cornell_movie_dialogs_lora/actions"><img alt="CI" src="https://img.shields.io/badge/CI-none-lightgrey"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6+-ee4c2c"></a>
  <a href="https://huggingface.co/docs/transformers/index"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-4.55.2-yellow"></a>
</p>

---

## ✨ What you get

- 🧩 **LoRA** fine-tuning (≈0.65% trainable params) on the Cornell corpus  
- ⚡ Ready for **RTX 4090** (BF16, optional grad checkpointing)  
- 🗂️ End-to-end scripts: data prep → train → inference  
- 🧪 Sample prompts + optional web demo (Gradio)

---

## 📦 Install

> **Pick ONE** Torch install, then install project deps.

**A) NVIDIA CUDA 12.4 (Linux / RTX 4090)**  
```bash
pip install "torch==2.6.0" --extra-index-url https://download.pytorch.org/whl/cu124
````

**B) CPU-only (Mac/Windows/Linux)**

```bash
pip install "torch==2.6.0"
```

**Then project deps**

```bash
pip install -r requirements.txt
```

💡 For the exact WSL+4090 environment, use the lock file:

```bash
pip install -r requirements-lock-wsl-cu124.txt
```

---

## 📂 Structure

```
.
├── build_pairs.py                 # raw Cornell → prompt/response pairs (JSONL)
├── split_pairs.py                 # train/validation/test splits
├── train_lora.py                  # LoRA training (PEFT + Transformers Trainer)
├── inference.py                   # interactive chat (CLI)
├── data/                          # processed JSONL (gitignored)
├── cornell-movie-dialogs-corpus/  # original dataset files (not required in repo)
├── out-cornell-phi3/              # LoRA adapter + tokenizer + (ignored) checkpoints
├── requirements.txt               # minimal portable deps
├── requirements-lock-wsl-cu124.txt# exact freeze used on 4090 box
└── train.log                      # training log (optional)
```

---

## 🧰 Data prep

1. Put the raw Cornell corpus here:

```
./cornell-movie-dialogs-corpus/
```

2. Build pairs & splits:

```bash
python build_pairs.py --data_dir cornell-movie-dialogs-corpus --out data/cornell_pairs.jsonl
python split_pairs.py --in data/cornell_pairs.jsonl \
  --train data/train.jsonl --valid data/validation.jsonl --test data/test.jsonl
```

---

## 🚀 Train (LoRA on RTX 4090)

```bash
python train_lora.py \
  --model_name_or_path microsoft/phi-3-mini-4k-instruct \
  --train_file data/train.jsonl \
  --validation_file data/validation.jsonl \
  --output_dir out-cornell-phi3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --bf16 \
  --gradient_checkpointing \
  --save_steps 500 --logging_steps 20
```

Resume after a crash:

```bash
# trainer.train(resume_from_checkpoint=True) is enabled in script
python train_lora.py ...
```

---

## 💬 Inference

```bash
python inference.py
# then type:
# You: We’re stranded on a desert highway. What now?
```

**Tip:** The script uses the model’s chat template when available for cleaner multi-turn behavior.

---

## 🔧 Optional: Merge LoRA → single model

If you want a single folder without PEFT at inference:

```bash
python - << 'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, os
base="microsoft/phi-3-mini-4k-instruct"
adapter="out-cornell-phi3"
out="out-cornell-phi3-merged"
tok=AutoTokenizer.from_pretrained(base)
m=AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
m=PeftModel.from_pretrained(m, adapter).merge_and_unload()
os.makedirs(out, exist_ok=True)
tok.save_pretrained(out)
m.save_pretrained(out, safe_serialization=True)
print("Merged ->", out)
PY
```

Then set `ADAPTER = "out-cornell-phi3-merged"` in `inference.py`.

---

## 🧪 Sample prompts

* *“I can’t believe you just did that! Do you know what this means?”*
* *“The car won’t start, and we’re in the middle of nowhere. What do we do?”*
* *“Don’t walk out that door—not after everything we’ve been through.”*
* *“You’re my grumpy older brother on a desert road trip. We just blew a tire.”*

Tune decoding:

* `temperature`: 0.7–0.9
* `top_p`: 0.9–0.95
* `repetition_penalty`: 1.05–1.2

---

## 📈 Latest training snapshot

```
Runtime: ~38m30s
Train loss: ~0.262
Samples/sec: ~258.6
Steps/sec: ~16.16
Epochs: 3
```

---

## 🧹 Git hygiene (large files)

This repo includes model artifacts. If you prefer smaller clones:

* Use **Git LFS** for `*.safetensors` / `*.pt` / `*.bin`, or
* Remove big files and publish adapters on the Hugging Face Hub.

---

## 📜 License

For research/education.
Check base model terms: [Phi-3 Mini Instruct](https://huggingface.co/microsoft/phi-3-mini-4k-instruct).

````

---

### Commit both

```bash
# write .gitignore and README.md as above, then:
git add .gitignore README.md
git commit -m "Clean .gitignore and polished README"
git push
````

Want me to also drop in a **Makefile** and an **examples script** (`samples.py`) so people can run end-to-end with single commands?
