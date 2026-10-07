#!/usr/bin/env python3
"""Run eval episodes in child processes and skip CUDA-crashed episodes.

CUDA illegal memory access poisons the current process, so recovery has to
happen from a parent process that launches fresh children.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


METRIC_RE = re.compile(
    r"InstanceImageNav succ/spl/dtg:\s*"
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)\((\d+)\)"
)

CUDA_CRASH_PATTERNS = (
    "CUDA error: an illegal memory access was encountered",
    "CUDA illegal memory access",
    "device-side assert triggered",
)

SUMMARY_EPISODE_NUMBERING = "official_habitat_episode_id"
DEFAULT_MANUAL_EXCLUSION_REASON = "manual exclusion requested"


def parse_episode_spec(spec):
    episodes = []
    seen = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start < 0 or end < start:
                raise argparse.ArgumentTypeError(
                    "invalid episode range: {}".format(part)
                )
            values = range(start, end + 1)
        else:
            value = int(part)
            if value < 0:
                raise argparse.ArgumentTypeError(
                    "episode must be non-negative: {}".format(part)
                )
            values = [value]
        for value in values:
            if value not in seen:
                seen.add(value)
                episodes.append(value)
    if not episodes:
        raise argparse.ArgumentTypeError("empty episode list")
    return episodes


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run main.py eval episodes one by one. If a child process crashes "
            "with a CUDA illegal-memory error, skip that episode and continue."
        )
    )
    parser.add_argument("--start", type=int, default=None,
                        help="official Habitat episode_id start, inclusive")
    parser.add_argument("--end", type=int, default=None,
                        help="official Habitat episode_id end, inclusive")
    parser.add_argument("--episodes", type=parse_episode_spec, default=None,
                        help="comma-separated Habitat episode_id list/ranges, e.g. 0-10,416-430")
    parser.add_argument(
        "--exclude-episodes",
        type=parse_episode_spec,
        default=None,
        help="comma-separated requested episode IDs/ranges to skip manually",
    )
    parser.add_argument(
        "--exclude-reason",
        default=DEFAULT_MANUAL_EXCLUSION_REASON,
        help="reason stored for episodes selected by --exclude-episodes",
    )
    parser.add_argument("--log-dir", required=True,
                        help="directory for per-episode logs and summary")
    parser.add_argument("--exp-name-prefix", default="autoskip_eval",
                        help="prefix for child --exp_name values")
    parser.add_argument("--summary-name", default="autoskip_summary.json",
                        help="summary JSON filename under --log-dir")
    parser.add_argument("--cuda-launch-blocking", action="store_true",
                        help="set CUDA_LAUNCH_BLOCKING=1 for child processes")
    parser.add_argument("--resume", action="store_true",
                        help="resume from an existing summary file")
    parser.add_argument("--dry-run", action="store_true",
                        help="print child commands without running them")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="child command after --, e.g. -- python -u main.py --eval 1 ...",
    )
    args = parser.parse_args()
    if args.episodes is None:
        if args.start is None or args.end is None:
            parser.error("provide either --episodes or both --start and --end")
        if args.start < 0 or args.end < args.start:
            parser.error("--start must be non-negative and --end must be >= --start")
        args.episode_list = list(range(args.start, args.end + 1))
    else:
        args.episode_list = list(args.episodes)
        if args.start is None:
            args.start = min(args.episode_list)
        if args.end is None:
            args.end = max(args.episode_list)
    args.exclude_episode_list = list(args.exclude_episodes or [])
    unknown_exclusions = sorted(
        set(args.exclude_episode_list) - set(args.episode_list)
    )
    if unknown_exclusions:
        parser.error(
            "excluded episode IDs are not in the requested episode list: {}".format(
                ",".join(str(value) for value in unknown_exclusions)
            )
        )
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing child command after --")
    return args


def remove_options(argv, names):
    cleaned = []
    skip_next = False
    names = set(names)
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        matched = False
        for name in names:
            if item == name:
                skip_next = True
                matched = True
                break
            if item.startswith(name + "="):
                matched = True
                break
        if not matched:
            cleaned.append(item)
    return cleaned


def build_child_command(base_cmd, habitat_episode_id, exp_name):
    cmd = remove_options(
        list(base_cmd),
        {
            "--eval_habitat_episode_id",
            "--eval_episode_start",
            "--eval_episode_end",
            "--num_eval_episodes",
            "--exp_name",
            "--eval_resume_log",
            "--eval_skip_episodes",
        },
    )
    cmd.extend([
        "--eval_habitat_episode_id", str(habitat_episode_id),
        "--num_eval_episodes", "1",
        "--exp_name", exp_name,
    ])
    return cmd


def parse_latest_metric(text):
    matches = list(METRIC_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    success = float(match.group(1))
    spl = float(match.group(2))
    dist = float(match.group(3))
    count = int(match.group(4))
    return {
        "success_rate": success,
        "spl": spl,
        "dist": dist,
        "count": count,
        "success_sum": success * count,
        "spl_sum": spl * count,
        "dist_sum": dist * count,
    }


def is_cuda_crash(text):
    return any(pattern in text for pattern in CUDA_CRASH_PATTERNS)


def load_summary(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def new_summary(args):
    return {
        "episode_numbering": SUMMARY_EPISODE_NUMBERING,
        "start": args.start,
        "end": args.end,
        "episodes": list(args.episode_list),
        "count": 0,
        "success_sum": 0.0,
        "spl_sum": 0.0,
        "dist_sum": 0.0,
        "completed": [],
        "skipped_cuda": [],
        "skipped_manual": [],
        "failed_other": [],
        "updated_at": None,
    }


def save_summary(path, summary):
    summary["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def add_metric(summary, episode, metric):
    count = int(metric["count"])
    summary["count"] += count
    summary["success_sum"] += float(metric["success_sum"])
    summary["spl_sum"] += float(metric["spl_sum"])
    summary["dist_sum"] += float(metric["dist_sum"])
    summary["completed"].append({
        "episode": int(episode),
        "count": count,
        "success_rate": float(metric["success_rate"]),
        "spl": float(metric["spl"]),
        "dist": float(metric["dist"]),
    })


def register_manual_exclusions(summary, episodes, reason):
    """Register requested exclusions and remove their stale failure records."""
    episode_set = {int(episode) for episode in episodes}
    skipped_manual = summary.setdefault("skipped_manual", [])
    registered = {int(item["episode"]) for item in skipped_manual}
    added = 0
    for episode in episodes:
        episode = int(episode)
        if episode not in registered:
            skipped_manual.append({
                "episode": episode,
                "reason": "manual exclusion: {}".format(reason),
            })
            registered.add(episode)
            added += 1

    failed_other = summary.setdefault("failed_other", [])
    retained_failures = [
        item for item in failed_other
        if int(item["episode"]) not in episode_set
    ]
    removed_failures = len(failed_other) - len(retained_failures)
    if removed_failures:
        summary["failed_other"] = retained_failures
    return added, removed_failures


def format_aggregate(summary):
    count = int(summary["count"])
    if count <= 0:
        return "InstanceImageNav succ/spl/dtg: nan/nan/nan(0)"
    return "InstanceImageNav succ/spl/dtg: {:.5f}/{:.5f}/{:.5f}({})".format(
        summary["success_sum"] / count,
        summary["spl_sum"] / count,
        summary["dist_sum"] / count,
        count,
    )


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / args.summary_name

    if args.resume:
        summary = load_summary(summary_path) or new_summary(args)
    else:
        summary = new_summary(args)

    episode_numbering = summary.get("episode_numbering")
    if episode_numbering != SUMMARY_EPISODE_NUMBERING:
        raise ValueError(
            "summary {} uses episode_numbering={!r}; expected {!r}. "
            "Do not mix legacy 1-based eval-slice summaries with official "
            "Habitat episode_id runs.".format(
                summary_path, episode_numbering, SUMMARY_EPISODE_NUMBERING
            )
        )

    added_exclusions, removed_failures = register_manual_exclusions(
        summary,
        args.exclude_episode_list,
        args.exclude_reason,
    )
    if (added_exclusions or removed_failures) and not args.dry_run:
        save_summary(summary_path, summary)

    done = {int(item["episode"]) for item in summary.get("completed", [])}
    skipped = {int(item["episode"]) for item in summary.get("skipped_cuda", [])}
    skipped_manual = {
        int(item["episode"]) for item in summary.get("skipped_manual", [])
    }

    env = os.environ.copy()
    if args.cuda_launch_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    for habitat_episode_id in args.episode_list:
        if habitat_episode_id in skipped_manual:
            print("[SKIP_EXCLUDED] habitat_episode_id={}".format(
                habitat_episode_id
            ), flush=True)
            continue
        if habitat_episode_id in done or habitat_episode_id in skipped:
            print("[SKIP_EXISTING] habitat_episode_id={}".format(habitat_episode_id), flush=True)
            continue

        exp_name = "{}_ep{:04d}".format(args.exp_name_prefix, habitat_episode_id)
        log_path = log_dir / "{}.log".format(exp_name)
        cmd = build_child_command(args.command, habitat_episode_id, exp_name)
        printable_cmd = " ".join(cmd)
        print("[RUN] habitat_episode_id={} log={} cmd={}".format(
            habitat_episode_id, log_path, printable_cmd
        ), flush=True)

        if args.dry_run:
            continue

        with log_path.open("wb") as log_file:
            header = (
                "# habitat_episode_id={}\n"
                "# episode_numbering={}\n"
                "# cmd={}\n"
                "# cuda_launch_blocking={}\n\n"
            ).format(
                habitat_episode_id,
                SUMMARY_EPISODE_NUMBERING,
                printable_cmd,
                int(bool(args.cuda_launch_blocking)),
            )
            log_file.write(header.encode("utf-8"))
            log_file.flush()
            proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                  env=env)

        text = log_path.read_text(errors="ignore")
        if proc.returncode == 0:
            metric = parse_latest_metric(text)
            if metric is None or int(metric["count"]) <= 0:
                summary["failed_other"].append({
                    "episode": int(habitat_episode_id),
                    "returncode": int(proc.returncode),
                    "reason": "missing eval metric",
                    "log": str(log_path),
                })
                save_summary(summary_path, summary)
                print("[ABORT] habitat_episode_id={} finished but no metric in {}".format(
                    habitat_episode_id, log_path
                ), flush=True)
                return 2
            add_metric(summary, habitat_episode_id, metric)
            save_summary(summary_path, summary)
            print("[DONE] habitat_episode_id={} {}".format(
                habitat_episode_id, format_aggregate(summary)
            ), flush=True)
            continue

        if is_cuda_crash(text):
            summary["skipped_cuda"].append({
                "episode": int(habitat_episode_id),
                "returncode": int(proc.returncode),
                "log": str(log_path),
            })
            save_summary(summary_path, summary)
            print("[CUDA_SKIP] habitat_episode_id={} returncode={} skipped_count={}".format(
                habitat_episode_id, proc.returncode, len(summary["skipped_cuda"])
            ), flush=True)
            continue

        summary["failed_other"].append({
            "episode": int(habitat_episode_id),
            "returncode": int(proc.returncode),
            "reason": "non-CUDA child failure",
            "log": str(log_path),
        })
        save_summary(summary_path, summary)
        print("[ABORT] habitat_episode_id={} returncode={} log={}".format(
            habitat_episode_id, proc.returncode, log_path
        ), flush=True)
        return int(proc.returncode) or 1

    print("[FINAL] {}".format(format_aggregate(summary)), flush=True)
    print("[FINAL] completed={} skipped_cuda={} skipped_manual={} failed_other={}".format(
        len(summary.get("completed", [])),
        len(summary.get("skipped_cuda", [])),
        len(summary.get("skipped_manual", [])),
        len(summary.get("failed_other", [])),
    ), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
