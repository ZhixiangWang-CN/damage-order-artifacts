#!/bin/bash
# Stop every addendum-8 worker on this machine, put unfinished jobs back in the queue, relaunch with run_v8.sh specs.
#   bash restart_v8.sh <run_v8 specs...>
# Finished conditions (runs/<model>/*.json.gz) are kept and skipped on resume. FAILED tracebacks are archived in runs/failed_log.
cd "$(dirname "$0")"
pkill -f '[a]tlas_full.py worker'
for i in $(seq 1 30); do pgrep -f '[a]tlas_full.py worker' > /dev/null || break; sleep 2; done
pgrep -f '[a]tlas_full.py worker' > /dev/null && { echo "workers still alive, aborting"; exit 1; }
mkdir -p runs/failed_log
n=0
for d in runs/claims/*/; do
  c=$(basename "$d")
  [ -f "$d/done" ] || [ -f "$d/SKIPPED" ] && continue
  if [[ "$c" == *__calib-* ]]; then
    m=${c%%__*}; k=${c#*__calib-}; k=${k%__0}
    [ -f "runs/$m/lock_$k.json" ] && continue
  fi
  [ -f "$d/FAILED" ] && cp "$d/FAILED" "runs/failed_log/${c}__archived_$(date +%s).txt"
  rm -rf "$d"; n=$((n+1))
done
rm -f runs/*/*.json.gz.tmp
echo "released $n claims"
bash run_v8.sh "$@"
