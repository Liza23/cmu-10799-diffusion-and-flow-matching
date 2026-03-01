#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run KID sweeps over guidance scales for:
  (a) unconditional generation
  (b) conditional generation with Blond_Hair

Outputs:
  - CSV summary: <output-root>/kid_guidance_summary.csv
  - LaTeX table: <output-root>/kid_guidance_table.tex
  - Per-run logs and generated images under <output-root>/w{1,3,5}/...

Usage:
  ./scripts/run_kid_guidance_table.sh \
    --checkpoint /path/to/checkpoint.pt \
    --method ddpm \
    [--dataset-path data/celeba-subset/train/images] \
    [--num-samples 1000] \
    [--batch-size 256] \
    [--num-steps 1000] \
    [--device cuda] \
    [--output-root logs/kid_guidance_YYYYmmdd_HHMMSS]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT=""
METHOD="ddpm"
DATASET_PATH="data/celeba-subset/train/images"
NUM_SAMPLES=1000
BATCH_SIZE=256
NUM_STEPS=""
DEVICE="cuda"
ATTR_NAME="Blond_Hair"
GUIDANCE_SCALES=(1 3 5)
OUTPUT_ROOT="logs/kid_guidance_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"; shift 2 ;;
    --method)
      METHOD="$2"; shift 2 ;;
    --dataset-path)
      DATASET_PATH="$2"; shift 2 ;;
    --num-samples)
      NUM_SAMPLES="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --num-steps)
      NUM_STEPS="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "Error: --checkpoint is required." >&2
  usage
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
RESULTS_CSV="$OUTPUT_ROOT/kid_guidance_summary.csv"
TABLE_TEX="$OUTPUT_ROOT/kid_guidance_table.tex"
echo "guidance_scale,mean_kid,blond_hair_kid" > "$RESULTS_CSV"

extract_kid_mean() {
  local log_path="$1"
  local kid_mean
  if command -v rg >/dev/null 2>&1; then
    kid_mean="$(rg -m1 "KID mean:" "$log_path" | awk '{print $3}')"
  else
    kid_mean="$(grep -m1 "KID mean:" "$log_path" | awk '{print $3}')"
  fi
  if [[ -z "${kid_mean:-}" ]]; then
    echo "Error: failed to parse KID mean from $log_path" >&2
    exit 1
  fi
  echo "$kid_mean"
}

run_eval() {
  local w="$1"
  local out_dir="$2"
  local log_path="$3"
  local attrs="$4"

  local cmd=(
    "$PYTHON_BIN" evaluate.py
    --checkpoint "$CHECKPOINT"
    --method "$METHOD"
    --dataset-path "$DATASET_PATH"
    --num-samples "$NUM_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --output-dir "$out_dir"
    --guidance "$w"
    --regenerate
  )

  if [[ -n "$NUM_STEPS" ]]; then
    cmd+=(--num-steps "$NUM_STEPS")
  fi
  if [[ -n "$attrs" ]]; then
    cmd+=(--attributes "$attrs")
  fi

  "${cmd[@]}" 2>&1 | tee "$log_path"
}

declare -A MEAN_KID
declare -A BLOND_KID

for w in "${GUIDANCE_SCALES[@]}"; do
  run_dir="$OUTPUT_ROOT/w${w}"
  mkdir -p "$run_dir"

  uncond_dir="$run_dir/unconditional_samples"
  uncond_log="$run_dir/unconditional_eval.log"
  echo "=== Running unconditional eval (w=$w) ==="
  run_eval "$w" "$uncond_dir" "$uncond_log" ""
  MEAN_KID["$w"]="$(extract_kid_mean "$uncond_log")"

  blond_dir="$run_dir/blond_hair_samples"
  blond_log="$run_dir/blond_hair_eval.log"
  echo "=== Running Blond_Hair conditional eval (w=$w) ==="
  run_eval "$w" "$blond_dir" "$blond_log" "$ATTR_NAME"
  BLOND_KID["$w"]="$(extract_kid_mean "$blond_log")"

  echo "$w,${MEAN_KID[$w]},${BLOND_KID[$w]}" >> "$RESULTS_CSV"
done

cat > "$TABLE_TEX" <<EOF
\\begin{table}[t]
\\centering
\\begin{tabular}{lcc}
\\toprule
\\textbf{Guidance Scale \$w\$} & \\textbf{Mean KID} & \\textbf{Blond\\_Hair KID} \\\\
\\midrule
1 & ${MEAN_KID[1]} & ${BLOND_KID[1]} \\\\
3 & ${MEAN_KID[3]} & ${BLOND_KID[3]} \\\\
5 & ${MEAN_KID[5]} & ${BLOND_KID[5]} \\\\
\\bottomrule
\\end{tabular}
\\caption{KID as a function of guidance scale \$w\$. Lower is better.}
\\end{table}
EOF

echo ""
echo "Saved CSV:   $RESULTS_CSV"
echo "Saved table: $TABLE_TEX"
