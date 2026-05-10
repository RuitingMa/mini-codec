#!/usr/bin/env bash
# Sequential train + eval for the W7 bitrate scan.
#
# 3 new training runs (1.6 / 6.4 / 12.8 kbps). The 3.2 kbps point is the
# existing baseline_train100 checkpoint, which we don't re-train. Each run
# is ~35 min of training + ~1 min of eval on a single 4090; total ~2 hours.
#
# Run from the repo root:
#     bash scripts/exp_d_run.sh
#
# Logs go to <out_dir>/tb (TensorBoard); every run writes a checkpoint at
# step 50000 and a per-sample CSV under <out_dir>/eval_ckpt_00050000/.
set -euo pipefail

cd "$(dirname "$0")/.."

# Sanity gates before sinking 2 hours of GPU time.
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA visible'"
git log --oneline -1
git status --short
echo

for KBPS in 1.6 6.4 12.8; do
    CFG="configs/exp_d_${KBPS}kbps.yaml"
    OUT="outputs/exp_d_${KBPS}kbps"
    CKPT="${OUT}/ckpt_00050000.pt"

    if [[ -f "${CKPT}" ]]; then
        echo "=== exp_d_${KBPS}kbps: checkpoint already exists, skipping training ==="
    else
        echo "=== exp_d_${KBPS}kbps: training (~35 min) ==="
        python -m src.train --config "${CFG}" --logger tensorboard
    fi

    if [[ -f "${OUT}/eval_ckpt_00050000/per_sample.csv" ]]; then
        echo "=== exp_d_${KBPS}kbps: eval CSV already exists, skipping eval ==="
    else
        echo "=== exp_d_${KBPS}kbps: eval on test-clean (256 utterances) ==="
        python scripts/eval.py --ckpt "${CKPT}" \
            --split test-clean --num-samples 256 --num-dump 32
    fi

    echo
done

echo "=== bitrate scan complete ==="
echo "Per-sample CSVs:"
ls -la outputs/exp_d_*kbps/eval_ckpt_00050000/per_sample.csv
echo
echo "Pack everything for scp:"
echo "    tar -czf w7_bitrate_scan.tar.gz \\"
echo "        outputs/baseline_train100/eval_ckpt_00050000 \\"
echo "        outputs/exp_d_*kbps/eval_ckpt_00050000"
