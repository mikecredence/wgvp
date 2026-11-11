#!/bin/bash
# (Maintainers, or any site whose GPU is a different machine.) Run ONLY the GPU stage of a data root on a remote
# NVIDIA box that has GVHMR installed. The CPU stages have been run locally in the pinned environment
# (`python -m wgvp run <root> --cpu-only`), which left the lifter inputs (cropped subject chunks + box tracks) under
# <root>/wgvp_output/<session>/work/<view>/<recording>/chunks/. This uploads the package and <root>/wgvp_output,
# runs `python -m wgvp gvhmr` there (GVHMR per chunk, assembly, post-GVHMR QC refresh, summary), pulls back the
# pose-level deliverables and the chunk records/logs, and ALWAYS stops the instance on exit.
# Environment: WGVP_EC2_ID, WGVP_SSH_HOST, WGVP_GVHMR_ROOT (default /home/ubuntu/GVHMR), WGVP_GVHMR_ENV (gvhmr),
# WGVP_REMOTE_DIR (/data/wgvp_gpu), AWS_PROFILE / AWS_REGION. See tools/remote_gpu_e2e.sh.
#   bash tools/remote_gpu_stage.sh <local data root>
set -eu
ROOT="$(cd "${1:?local data root}" && pwd)"
PKG="$(cd "$(dirname "$0")/.." && pwd)"
ID="${WGVP_EC2_ID:?set WGVP_EC2_ID}"; H="${WGVP_SSH_HOST:?set WGVP_SSH_HOST}"
GROOT="${WGVP_GVHMR_ROOT:-/home/ubuntu/GVHMR}"; GENV="${WGVP_GVHMR_ENV:-gvhmr}"; B="${WGVP_REMOTE_DIR:-/data/wgvp_gpu}"
AWS="aws ${AWS_PROFILE:+--profile $AWS_PROFILE} ${AWS_REGION:+--region $AWS_REGION}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 $H"
stop_box() { echo "[trap] stopping $ID"; $AWS ec2 stop-instances --instance-ids "$ID" --output text >/dev/null || true; $AWS ec2 wait instance-stopped --instance-ids "$ID" || true
             echo "instance state: $($AWS ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text)"; }
trap stop_box EXIT
N=$(find "$ROOT/wgvp_output" -path '*/chunks/chunk_*.json' | wc -l | tr -d ' '); [ "$N" -gt 0 ] || { echo "no lifter chunks under $ROOT/wgvp_output: run the CPU stages first"; exit 1; }
echo "$N chunk record(s) to lift"
STATE=$($AWS ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text); echo "instance state: $STATE"
if [ "$STATE" != "running" ]; then $AWS ec2 start-instances --instance-ids "$ID" --output text >/dev/null; $AWS ec2 wait instance-running --instance-ids "$ID"; sleep 45; fi
until $SSH true 2>/dev/null; do sleep 15; done
GPU=$($SSH 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader,nounits'); echo "GPU: $GPU"
MEM=$(echo "$GPU" | awk -F', ' '{print $3}'); [ "$MEM" -lt 1000 ] || { echo "GPU busy ($MEM MiB), aborting"; exit 1; }
$SSH "rm -rf $B && mkdir -p $B/root"
rsync -a --exclude '.git' --exclude '__pycache__' --exclude 'examples' --exclude 'docs' --exclude 'sample' --exclude 'sample_v*' --exclude 'wgvp-env' --exclude '*.egg-info' --exclude 'wgvp_output' "$PKG/" "$H:$B/pkg/"
rsync -a "$ROOT/wgvp_config.json" "$H:$B/root/" 2>/dev/null || true
rsync -a --exclude 'gvhmr/' "$ROOT/wgvp_output/" "$H:$B/root/wgvp_output/"
echo "uploaded $(du -sh "$ROOT/wgvp_output" | cut -f1) of CPU-stage outputs"
ACT="source ~/miniconda3/etc/profile.d/conda.sh; conda activate $GENV; cd $B/pkg"
$SSH "cd $B && (setsid nohup bash -c '$ACT; python -m wgvp gvhmr $B/root --gvhmr-root $GROOT --gvhmr-python \$(which python) > $B/gvhmr.log 2>&1; echo done > $B/all.done' > $B/nohup.log 2>&1 < /dev/null &)"
echo "launched at $(date); waiting ..."
T0=$(date +%s)
until $SSH "[ -f $B/all.done ]" 2>/dev/null; do
  sleep 60; echo "  $(( ($(date +%s) - T0) / 60 )) min: $($SSH "tail -1 $B/gvhmr.log 2>/dev/null" 2>/dev/null | cut -c1-150)"
  [ $(( $(date +%s) - T0 )) -lt 10800 ] || { echo "timeout after 3 h"; break; }
done
# back: the deliverables (pose-level only), the summary, and the chunk records / GVHMR logs (text)
rsync -a --exclude 'work' "$H:$B/root/wgvp_output/" "$ROOT/wgvp_output/"
rsync -a --include '*/' --include 'gvhmr_run.log' --include 'chunk_*.json' --exclude '*' "$H:$B/root/wgvp_output/" "$ROOT/wgvp_output/"
rsync -a "$H:$B/gvhmr.log" "$H:$B/nohup.log" "$ROOT/wgvp_output/logs/" || true
$SSH "nvidia-smi --query-gpu=memory.used --format=csv,noheader" || true
echo "GPU stage pulled into $ROOT/wgvp_output at $(date)"
