#!/bin/bash
# (Maintainers.) End-to-end run WITH the GPU stage on a remote NVIDIA box that already has GVHMR installed, then pull
# the pose-level outputs back and stop the instance. Nothing about a particular machine is hard-coded: set
#   WGVP_EC2_ID        the EC2 instance id (started if stopped, ALWAYS stopped on exit, even on error)
#   WGVP_SSH_HOST      an ssh alias for it (key-based login, user with conda + GVHMR)
#   WGVP_GVHMR_ROOT    GVHMR checkout on the box            (default /home/ubuntu/GVHMR)
#   WGVP_GVHMR_ENV     conda env holding GVHMR             (default gvhmr)
#   WGVP_REMOTE_DIR    scratch dir on the box               (default /data/wgvp_e2e)
#   AWS_PROFILE / AWS_REGION for the aws cli (start / stop / state)
# What runs on the box, inside the GVHMR env (its python also runs the CPU stages; `lap` is installed if missing):
#   1. check-gpu
#   2. the dry run on sample/ (HC900) with the GPU stage: validates panels, selection, sync, chunking at exits,
#      GVHMR, reconciliation against the sample's known answers
#   3. the pipeline on sample_v2/data (HC901, two compromised views) with the GPU stage, for tools/two_view_fusion.py
#   bash tools/remote_gpu_e2e.sh <local output dir>
set -eu
OUT="${1:?local output dir}"; mkdir -p "$OUT"
PKG="$(cd "$(dirname "$0")/.." && pwd)"
ID="${WGVP_EC2_ID:?set WGVP_EC2_ID}"; H="${WGVP_SSH_HOST:?set WGVP_SSH_HOST}"
GROOT="${WGVP_GVHMR_ROOT:-/home/ubuntu/GVHMR}"; GENV="${WGVP_GVHMR_ENV:-gvhmr}"; B="${WGVP_REMOTE_DIR:-/data/wgvp_e2e}"
AWS="aws ${AWS_PROFILE:+--profile $AWS_PROFILE} ${AWS_REGION:+--region $AWS_REGION}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 $H"
stop_box() { echo "[trap] stopping $ID"; $AWS ec2 stop-instances --instance-ids "$ID" --output text >/dev/null || true; $AWS ec2 wait instance-stopped --instance-ids "$ID" || true
             echo "instance state: $($AWS ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text)"; }
trap stop_box EXIT
STATE=$($AWS ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text); echo "instance state: $STATE"
if [ "$STATE" != "running" ]; then $AWS ec2 start-instances --instance-ids "$ID" --output text >/dev/null; $AWS ec2 wait instance-running --instance-ids "$ID"; sleep 45; fi
until $SSH true 2>/dev/null; do sleep 15; done
GPU=$($SSH 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader,nounits'); echo "GPU: $GPU"
MEM=$(echo "$GPU" | awk -F', ' '{print $3}'); [ "$MEM" -lt 1000 ] || { echo "GPU busy ($MEM MiB), aborting"; exit 1; }
$SSH "rm -rf $B && mkdir -p $B"
rsync -a --exclude '.git' --exclude '__pycache__' --exclude 'examples/bundled' --exclude 'examples/data' --exclude 'examples/_cache' --exclude 'docs/figures' --exclude 'wgvp_output' "$PKG/" "$H:$B/pkg/"
echo "uploaded $(du -sh --exclude .git --exclude examples --exclude docs "$PKG" 2>/dev/null | cut -f1) package"
ACT="source ~/miniconda3/etc/profile.d/conda.sh; conda activate $GENV; cd $B/pkg"
$SSH "$ACT; python -c 'import lap' 2>/dev/null && echo 'lap present' || { echo 'lap missing: installing'; pip install -q lap==0.5.13 2>&1 | tail -1; }" | tee "$OUT/lap_install.txt"
# everything nohup'd under setsid so an ssh drop cannot kill it; a marker file says when it is finished
$SSH "cd $B && (setsid nohup bash -c '$ACT; python -m wgvp check-gpu --gvhmr-root $GROOT > $B/check_gpu.txt 2>&1;
  python -m wgvp dry-run --out $B/dryrun --gvhmr-root $GROOT --gvhmr-python \$(which python) > $B/dryrun.log 2>&1;
  mkdir -p $B/v2 && cp -r sample_v2/data/. $B/v2/ && python -m wgvp run $B/v2 --gvhmr-root $GROOT --gvhmr-python \$(which python) > $B/v2.log 2>&1;
  echo done > $B/all.done' > $B/nohup.log 2>&1 < /dev/null &)"
echo "launched at $(date); waiting ..."
T0=$(date +%s)
until $SSH "[ -f $B/all.done ]" 2>/dev/null; do
  sleep 60; echo "  $(( ($(date +%s) - T0) / 60 )) min: $($SSH "tail -1 $B/v2.log 2>/dev/null || tail -1 $B/dryrun.log 2>/dev/null" 2>/dev/null | cut -c1-150)"
  [ $(( $(date +%s) - T0 )) -lt 10800 ] || { echo "timeout after 3 h"; break; }
done
for R in dryrun v2; do
  rsync -a --exclude 'work' "$H:$B/$R/wgvp_output/" "$OUT/$R/wgvp_output/" || true
  rsync -a --include '*/' --include 'gvhmr_run.log' --include 'chunk_*.json' --exclude '*' "$H:$B/$R/wgvp_output/" "$OUT/$R/work_logs/" || true
done
rsync -a "$H:$B/dryrun.log" "$H:$B/v2.log" "$H:$B/check_gpu.txt" "$H:$B/nohup.log" "$OUT/" || true
$SSH "nvidia-smi --query-gpu=memory.used --format=csv,noheader" || true
echo "pulled to $OUT at $(date)"
