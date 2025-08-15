Complete **`README.md`** for your Cornell Movie Dialogs fine-tuning project, based on everything done so far:

---

```markdown
# Cornell Movie Dialogs Fine-Tuning with LoRA (Phi-3)

This project fine-tunes Microsoft's **Phi-3 Mini (4K instruct)** model using **LoRA (Low-Rank Adaptation)** on the **Cornell Movie Dialogs Corpus**.  
The goal is to teach the model conversational patterns and style found in movie scripts, enabling it to produce more natural, character-like responses.

---

## 📌 Goals

- Fine-tune a large language model (Phi-3 Mini) on real movie conversation data.
- Use **LoRA** for efficient training without updating all model parameters.
- Build an **inference script** for interactive chat with the fine-tuned model.
- Prepare the project for **reproducible training** and **GitHub release**.

---

## 📂 Dataset

**Cornell Movie Dialogs Corpus** contains:
- 220,579 conversational exchanges between 10,292 pairs of movie characters.
- Metadata about characters, movies, and genres.
- Rich linguistic variety in informal, movie-style dialogue.

We preprocess this dataset into train, validation, and test splits.

---

## ⚙️ Technology Stack

- **Python 3.11**
- **PyTorch** with CUDA 12.4 acceleration
- **Hugging Face Transformers**
- **PEFT (Parameter-Efficient Fine-Tuning)**
- **BitsAndBytes** (for quantization and memory efficiency)
- **Datasets** library for handling JSONL data

---

## 🧠 Algorithms & Approach

1. **Base Model**  
   - `microsoft/phi-3-mini-4k-instruct`  
   - Small, instruction-tuned LLM optimized for reasoning and low-latency inference.

2. **LoRA Fine-Tuning**  
   - Only a small fraction of model weights are updated.
   - Saves memory, speeds up training.
   - Gradient checkpointing used to reduce GPU memory usage.

3. **Data Preprocessing**  
   - `build_pairs.py` creates paired conversation lines.
   - `split_pairs.py` splits into training, validation, and test sets.

4. **Training**  
   - `train_lora.py` runs LoRA fine-tuning with Hugging Face `Trainer`.
   - Checkpoints saved every N steps.

5. **Inference**  
   - `inference.py` loads the base model and LoRA adapter.
   - Interactive CLI chat loop.

---

## 📁 Project Structure

```

cornell\_movie\_dialogs/
├── build\_pairs.py                  # Create paired dialogues from raw dataset
├── split\_pairs.py                   # Split into train/val/test sets
├── train\_lora.py                    # LoRA fine-tuning script
├── inference.py                     # Chat with the fine-tuned model
├── train.log                        # Training log
├── data/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   └── cornell\_pairs.jsonl
├── cornell-movie-dialogs-corpus/    # Original dataset files
├── out-cornell-phi3/                 # Fine-tuned model output
│   ├── adapter\_model.safetensors
│   ├── adapter\_config.json
│   ├── tokenizer files
│   ├── checkpoint-\* directories
│   └── training\_args.bin
└── requirements.txt                 # Exact dependencies

````

---

## 🚀 Quick Start

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
````

### 2️⃣ Prepare dataset

```bash
python build_pairs.py
python split_pairs.py
```

### 3️⃣ Train model

```bash
python train_lora.py
```

### 4️⃣ Run inference

```bash
python inference.py
You: Hello there!
Bot: Hi! How are you doing today?
```

---

## 🖥️ Training Details

* **GPU**: NVIDIA RTX 4090 (24 GB VRAM)
* **Batch size**: Tuned for GPU capacity
* **Optimizer**: AdamW
* **Learning rate**: \~1e-5
* **LoRA rank**: Set for efficiency
* **Epochs**: 3
* **Final loss**: \~0.26

---

## 📊 Example Outputs

**Prompt:**

```
You: The car won’t start, and we’re in the middle of nowhere. What do we do?
```

**Model:**

```
Bot: Well, I guess we’d better get out of here.
```

---

## 🔮 Next Steps

* Merge LoRA adapter into a single model file (`out-cornell-phi3-merged`) for easier deployment.
* Improve prompt templates for more engaging responses.
* Fine-tune on additional conversational datasets for broader knowledge.
* Deploy as a web app with Gradio.

---

## 📜 License

This project uses the Cornell Movie Dialogs Corpus (research license)
and Microsoft Phi-3 Mini (model license from Hugging Face).

---

## 🤝 Acknowledgments

* Hugging Face Transformers & Datasets
* Microsoft for Phi-3 model
* Cornell University for the movie dialogues corpus

```

---

Do you want me to also **zip this README.md with `requirements.txt`** so you have a ready-to-upload GitHub starter? That would make the repo immediately publishable.
```
