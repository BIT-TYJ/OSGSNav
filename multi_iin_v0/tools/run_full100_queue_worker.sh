#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GPU_ID" >&2
  exit 2
fi

task_gpu="$1"
task_repo="${MULTI_IIN_REPO:-/nas_ds925/tyj/IEVE}"
task_root="${MULTI_IIN_TASK_ROOT:-${task_repo}/multi_iin_v0/results/experiments/full100_compact_correctness_20260908}"
task_episode_count="${MULTI_IIN_EPISODE_COUNT:-100}"
task_python="${MULTI_IIN_PYTHON:-/code1/tyj/anaconda3/envs/ieve/bin/python}"
task_worker="${task_repo}/multi_iin_v0/tools/run_full100_worker.sh"
task_claims="${task_root}/.episode_claims"
task_log="${task_root}/worker_logs/queue_gpu${task_gpu}.log"

mkdir -p "${task_claims}" "${task_root}/worker_logs"

episode_complete_any() {
  local episode_index="$1"
  "${task_python}" -c \
    'import glob,json,sys
root,episode=sys.argv[1],int(sys.argv[2])
paths=glob.glob(f"{root}/ep{episode:03d}_gpu*/full_result.json")
for path in paths:
    try:
        data=json.load(open(path))
        episodes=data.get("episodes", [])
        if len(episodes)==1 and len(episodes[0].get("goals", []))==2:
            raise SystemExit(0)
    except Exception:
        pass
raise SystemExit(1)' \
    "${task_root}" "${episode_index}" >/dev/null 2>&1
}

current_claim=""
release_claim() {
  if [[ -n "${current_claim}" ]]; then
    rmdir "${current_claim}" 2>/dev/null || true
    current_claim=""
  fi
}
trap release_claim EXIT INT TERM

while true; do
  claimed_episode=""
  for ((episode_index=0; episode_index<task_episode_count; episode_index+=1)); do
    if episode_complete_any "${episode_index}"; then
      continue
    fi
    claim_dir="${task_claims}/ep$(printf '%03d' "${episode_index}")"
    if mkdir "${claim_dir}" 2>/dev/null; then
      current_claim="${claim_dir}"
      if episode_complete_any "${episode_index}"; then
        release_claim
        continue
      fi
      claimed_episode="${episode_index}"
      break
    fi
  done

  if [[ -z "${claimed_episode}" ]]; then
    completed="$(${task_python} -c \
      'import glob,json,sys
done=set()
for path in glob.glob(sys.argv[1]+"/ep*_gpu*/full_result.json"):
    try:
        data=json.load(open(path)); episodes=data.get("episodes", [])
        if len(episodes)==1 and len(episodes[0].get("goals", []))==2:
            done.add(episodes[0].get("multi_episode_id"))
    except Exception:
        pass
print(len(done))' "${task_root}")"
    if (( completed >= task_episode_count )); then
      echo "QUEUE gpu=${task_gpu} complete ${completed}/${task_episode_count} $(date '+%F %T')" | tee -a "${task_log}"
      exit 0
    fi
    echo "QUEUE gpu=${task_gpu} waiting: completed=${completed}/${task_episode_count}, remaining work claimed $(date '+%F %T')" | tee -a "${task_log}"
    sleep 30
    continue
  fi

  echo "CLAIM gpu=${task_gpu} episode=${claimed_episode} $(date '+%F %T')" | tee -a "${task_log}"
  "${task_worker}" "${task_gpu}" "${claimed_episode}" >>"${task_log}" 2>&1
  worker_code=$?
  if episode_complete_any "${claimed_episode}"; then
    echo "RELEASE gpu=${task_gpu} episode=${claimed_episode} complete $(date '+%F %T')" | tee -a "${task_log}"
  else
    echo "RELEASE gpu=${task_gpu} episode=${claimed_episode} incomplete worker_exit=${worker_code} $(date '+%F %T')" | tee -a "${task_log}"
  fi
  release_claim
  if (( worker_code != 0 )); then
    sleep 30
  fi
done
