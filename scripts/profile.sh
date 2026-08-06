#!/bin/bash
# Set a serving profile in .env.dspark, then start/stop the 2-node cluster.
# Usage: ./profile.sh set TP_SIZE=2 PP_SIZE=1 MAX_NUM_SEQS=48 ...
#        ./profile.sh up | down | show
set -euo pipefail
ENV_FILE=/opt/dsv4/recipe/.env.dspark
REPO=/opt/dsv4/recipe
WORKER=WORKER_ROCE_IP

case "${1:-show}" in
  set)
    shift
    for kv in "$@"; do
      k="${kv%%=*}"; v="${kv#*=}"
      if grep -qE "^${k}=" "$ENV_FILE"; then
        sed -i "s|^${k}=.*|${k}=${v}|" "$ENV_FILE"
      else
        echo "${k}=${v}" >> "$ENV_FILE"
      fi
      echo "  ${k}=${v}"
    done
    rsync -a "$ENV_FILE" "$WORKER:$ENV_FILE"
    ;;
  show)
    grep -E "^(TP_SIZE|PP_SIZE|SPEC_DECODE|MAX_MODEL_LEN|MAX_NUM_SEQS|MAX_NUM_BATCHED_TOKENS|GPU_MEMORY_UTILIZATION|MTP_NUM_TOKENS|DEFAULT_THINKING)=" "$ENV_FILE"
    ;;
  up)
    cd "$REPO" && ./start-deepseek-v4-flash-dspark.sh
    ;;
  down)
    cd "$REPO" && ./stop-deepseek-v4-flash-dspark.sh || true
    # hard teardown: orphaned workers have starved later launches on this cluster
    sudo docker ps -q --filter "ancestor=ghcr.io/anemll/dspark-vllm-gx10:0.1.1" | xargs -r sudo docker rm -f
    ssh "$WORKER" 'sudo docker ps -q --filter "ancestor=ghcr.io/anemll/dspark-vllm-gx10:0.1.1" | xargs -r sudo docker rm -f'
    sleep 5
    echo "spark-1 free: $(free -g | awk '/^Mem:/{print $7}')GB   spark-2 free: $(ssh $WORKER "free -g | awk '/^Mem:/{print \$7}'")GB"
    ;;
esac
