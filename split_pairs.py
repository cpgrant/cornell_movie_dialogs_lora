#!/usr/bin/env python
# split_pairs.py
import argparse, json, random, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/cornell_pairs.jsonl")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--valid", default="data/validation.jsonl")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--valid_size", type=float, default=0.05)
    ap.add_argument("--test_size", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.train), exist_ok=True)

    data = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    random.seed(args.seed)
    random.shuffle(data)

    n = len(data)
    n_test = int(n * args.test_size)
    n_valid = int(n * args.valid_size)
    test = data[:n_test]
    valid = data[n_test:n_test+n_valid]
    train = data[n_test+n_valid:]

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as out:
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(args.train, train)
    dump(args.valid, valid)
    dump(args.test, test)

    print(f"train={len(train)} valid={len(valid)} test={len(test)} (total={n})")

if __name__ == "__main__":
    main()
