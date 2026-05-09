#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

logs=(ucsf_task_*_gpu_*.log)
if [[ ${#logs[@]} -eq 0 ]]; then
  echo "No UCSF task logs found in $(pwd)" >&2
  exit 1
fi

printf "%-8s %-6s %s\n" "TASK" "GPU" "PROGRESS"
printf "%-8s %-6s %s\n" "----" "---" "--------"

for log_path in "${logs[@]}"; do
  name="${log_path%.log}"
  name="${name#ucsf_task_}"
  task_id="${name%%_gpu_*}"
  gpu_id="${name##*_gpu_}"

  progress=$(
    tr '\r' '\n' < "${log_path}" \
      | awk '/Single-mosaic inference:/ { line = $0 } END { print line }'
  )
  progress="${progress%%\[INFO*}"
  progress="${progress%"${progress##*[![:space:]]}"}"

  if [[ -z "${progress}" ]]; then
    progress="no progress line yet"
  fi

  printf "%-8s %-6s %s\n" "${task_id}" "${gpu_id}" "${progress}"
done
