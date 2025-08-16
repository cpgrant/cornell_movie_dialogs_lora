#!/usr/bin/env bash
# scripts/audit_paths.sh
set -euo pipefail

command -v rg >/dev/null || { echo "ripgrep (rg) required"; exit 2; }

# in scripts/audit_paths.sh, add to RG_BASE_ARGS:
RG_BASE_ARGS=( --line-number --hidden --glob '!outputs/**' --glob '!.git/**' --glob '!scripts/**' )


section () {
  printf "\n\033[1m== %s ==\033[0m\n" "$1"
  shift
  if "$@"; then :; else echo "(none)"; fi
}

# 1) References to outputs/
section "References to outputs/ (outside outputs/ dir)" \
  rg "${RG_BASE_ARGS[@]}" -S -e "outputs/" .

# 2) Legacy paths (ok in archive/docs, otherwise fix)
section "OLD paths: out-cornell-phi3" \
  rg "${RG_BASE_ARGS[@]}" -S -e "out-cornell-phi3" .

section "OLD paths: out-cornell-phi3-merged" \
  rg "${RG_BASE_ARGS[@]}" -S -e "out-cornell-phi3-merged" .

# 3) Writers
section "Writers (open(...,'w'/'a'), json.dump, torch.save, save_model/pretrained, to_json/to_csv)" \
  rg "${RG_BASE_ARGS[@]}" \
     -e "open\([^)]*,\s*['\"]w" \
     -e "open\([^)]*,\s*['\"]a" \
     -e "json\.dump" \
     -e "torch\.save" \
     -e "save_model" \
     -e "save_pretrained" \
     -e "to_json" \
     -e "to_csv" src/ archive/ ./*.py

# 4) Readers
section "Readers (open(...,'r'), json.load, torch.load, datasets.load_dataset, from_pretrained)" \
  rg "${RG_BASE_ARGS[@]}" \
     -e "open\([^)]*,\s*['\"]r" \
     -e "json\.load" \
     -e "torch\.load" \
     -e "load_dataset\(" \
     -e "from_pretrained\(" src/ archive/ ./*.py

# 5) Data paths
section "Data paths mentioning data/" \
  rg "${RG_BASE_ARGS[@]}" -e "data/" src/ archive/ ./*.py

# 6) Logs / eval outputs
section "Mentions of logs/ or eval_results/ or train.log" \
  rg "${RG_BASE_ARGS[@]}" -e "train\.log" -e "eval_results" -e "logs/" .

# 7) argparse defaults for common I/O params
section "argparse defaults for train/validation/test/prompts/out/output_dir/model_dir/adapter" \
  rg "${RG_BASE_ARGS[@]}" \
     -e "add_argument\(--train[^)]*default=" \
     -e "add_argument\(--validation[^)]*default=" \
     -e "add_argument\(--test[^)]*default=" \
     -e "add_argument\(--prompts[^)]*default=" \
     -e "add_argument\(--out[^)]*default=" \
     -e "add_argument\(--output_dir[^)]*default=" \
     -e "add_argument\(--model_dir[^)]*default=" \
     -e "add_argument\(--adapter[^)]*default=" src/ ./*.py

# 8) Bare-filename WRITES (no directory)
section "Bare-filename WRITES (no directory in open(...,'w'))" \
  rg "${RG_BASE_ARGS[@]}" -e "open\(\s*['\"][^/ \t]['\"][^,]*,\s*['\"]w" src/ archive/ ./*.py

# 9) Absolute paths in quoted strings (simple heuristic; exclude URLs)
section "Absolute paths inside quoted strings (excluding URLs)" \
  bash -lc 'rg --line-number --hidden --glob "!outputs/**" --glob "!.git/**" -e "\"/" -e "'\''/" src/ archive/ ./*.py | rg -v "https?://" || true'
