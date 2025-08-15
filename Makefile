setup:
\tpip install -r requirements.txt

pairs:
\tpython build_pairs.py --data_dir cornell-movie-dialogs-corpus --out data/cornell_pairs.jsonl

split:
\tpython split_pairs.py --in data/cornell_pairs.jsonl --train data/train.jsonl --valid data/validation.jsonl --test data/test.jsonl

train:
\tpython train_lora.py --model_name_or_path microsoft/phi-3-mini-4k-instruct --train_file data/train.jsonl --validation_file data/validation.jsonl --output_dir out-cornell-phi3 --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --learning_rate 2e-4 --num_train_epochs 3 --bf16 --gradient_checkpointing --save_steps 500 --logging_steps 20

infer:
\tpython inference.py
