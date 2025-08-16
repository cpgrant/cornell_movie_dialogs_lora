.PHONY: setup train merge eval infer

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip -r requirements.txt

train:
	. .venv/bin/activate && python train_lora.py

merge:
	. .venv/bin/activate && python -m peft.utils.merge_adapter \
	  --base microsoft/phi-3-mini-4k-instruct \
	  --adapter outputs/adapters/phi3-cornell-lora \
	  --out outputs/merged/phi3-cornell-merged-latest

eval:
	. .venv/bin/activate && python src/eval/ab_eval.py

infer:
	. .venv/bin/activate && python inference.py --model_dir outputs/merged/phi3-cornell-merged-latest
