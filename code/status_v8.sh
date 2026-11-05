#!/bin/bash
# progress of the addendum-8 run on this machine.  bash status_v8.sh
cd "$(dirname "$0")"
echo "workers alive: $(ps aux | grep -c '[a]tlas_full.py worker')"
for m in q15 llama phi q7 mis gemma q14; do
  [ -d runs/$m ] || continue
  echo "$m: conditions $(ls runs/$m/*.json.gz 2>/dev/null | wc -l) | locks $(ls runs/$m/lock_*.json 2>/dev/null | sed 's/.*lock_//;s/.json//' | tr '\n' ' ')| usable $( [ -f runs/$m/usable.json ] && python3 -c "import json;u=json.load(open('runs/$m/usable.json'));print(len(u['usable']),'measures,',len(u['tabled_tasks']),'tasks')")"
done
echo "jobs done $(ls -d runs/claims/*/ 2>/dev/null | while read d; do [ -f $d/done ] && echo x; done | wc -l) / claimed $(ls -d runs/claims/*/ 2>/dev/null | grep -vc calib-) | FAILED: $(ls runs/claims/*/FAILED 2>/dev/null | wc -l)"
grep -h "Error\|Traceback" logs/worker_*.log 2>/dev/null | sort | uniq -c | head -5
