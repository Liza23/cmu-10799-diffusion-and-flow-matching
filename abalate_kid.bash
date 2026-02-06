# --- EDIT THESE ---
FM_CKPT="logs/flow_matching_20260201_060908/checkpoints/flow_matching_0100000.pt"
DDPM_CKPT="logs/ddpm_20260201_212111/checkpoints/ddpm_final.pt"
DATASET="data/celeba-subset/train/images"

# steps to sweep
STEPS="1 5 10 50 100 200 1000"

RESULTS_DIR="logs/kid_results"
mkdir -p "$RESULTS_DIR"
FM_RESULTS="$RESULTS_DIR/flow_matching_kid.txt"
DDIM_RESULTS="$RESULTS_DIR/ddim_kid.txt"
echo "step kid_mean kid_std" > "$FM_RESULTS"
echo "step kid_mean kid_std" > "$DDIM_RESULTS"

# --- FLOW MATCHING ---
for s in $STEPS; do
  OUT="logs/flow_matching_20260201_060908/checkpoints/samples/fm_${s}"
  python sample.py \
    --checkpoint "$FM_CKPT" \
    --method flow_matching \
    --num_steps "$s" \
    --num_samples 1000 \
    --batch_size 256 \
    --output_dir "$OUT"

  eval_out=$(python evaluate.py \
    --checkpoint "$FM_CKPT" \
    --method flow_matching \
    --dataset-path "$DATASET" \
    --output-dir "$OUT")

  printf "%s\n" "$eval_out" > "$OUT/kid_eval.txt"
  kid_mean=$(printf "%s\n" "$eval_out" | rg -m1 "KID mean:" | awk '{print $3}')
  kid_std=$(printf "%s\n" "$eval_out" | rg -m1 "KID std:" | awk '{print $3}')
  echo "$s $kid_mean $kid_std" >> "$FM_RESULTS"
done

# --- DDIM (DDPM checkpoint) ---
for s in $STEPS; do
  OUT="logs/ddpm_20260201_212111/checkpoints/samples/ddim_${s}"
  python sample.py \
    --checkpoint "$DDPM_CKPT" \
    --method ddpm \
    --sampler ddim \
    --num_steps "$s" \
    --num_samples 1000 \
    --batch_size 256 \
    --output_dir "$OUT"

  eval_out=$(python evaluate.py \
    --checkpoint "$DDPM_CKPT" \
    --method ddpm \
    --dataset-path "$DATASET" \
    --output-dir "$OUT")

  printf "%s\n" "$eval_out" > "$OUT/kid_eval.txt"
  kid_mean=$(printf "%s\n" "$eval_out" | rg -m1 "KID mean:" | awk '{print $3}')
  kid_std=$(printf "%s\n" "$eval_out" | rg -m1 "KID std:" | awk '{print $3}')
  echo "$s $kid_mean $kid_std" >> "$DDIM_RESULTS"
done
