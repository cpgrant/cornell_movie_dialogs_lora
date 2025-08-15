#!/usr/bin/env python
# build_pairs.py
import argparse, json, os, re

def load_movie_lines(lines_path):
    # Cornell format: lineID +++$+++ characterID +++$+++ movieID +++$+++ character +++$+++ text
    lines = {}
    with open(lines_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) >= 5:
                line_id, text = parts[0], parts[4]
                lines[line_id] = text
    return lines

def load_conversations(convs_path):
    # Format: characterID +++$+++ characterID +++$+++ movieID +++$+++ ['L1045','L1044','L985',...]
    convs = []
    with open(convs_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) >= 4:
                ids_str = parts[3]
                ids = re.findall(r"L\d+", ids_str)
                if ids:
                    convs.append(ids)
    return convs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="cornell-movie-dialogs-corpus", help="Folder with movie_lines.txt and movie_conversations.txt")
    ap.add_argument("--out", default="data/cornell_pairs.jsonl", help="Output JSONL of prompt/response pairs")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    lines_path = os.path.join(args.data_dir, "movie_lines.txt")
    convs_path = os.path.join(args.data_dir, "movie_conversations.txt")

    lines = load_movie_lines(lines_path)
    convs = load_conversations(convs_path)

    count = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for ids in convs:
            for i in range(len(ids) - 1):
                a = lines.get(ids[i], "").strip()
                b = lines.get(ids[i+1], "").strip()
                if a and b:
                    # save as simple dialogue pair
                    rec = {"prompt": a, "response": b}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
    print(f"Wrote {count} pairs to {args.out}")

if __name__ == "__main__":
    main()
