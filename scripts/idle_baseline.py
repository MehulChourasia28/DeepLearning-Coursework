#!/usr/bin/env python3
"""Measure this machine's idle power (fixed background draw) with the group run logger.

CodeCarbon's machine mode measures the whole computer, so every run includes the idle desktop. This
record lets the report separate that fixed draw from model computation (scaling analysis, idle share).
Close other heavy programs first and leave the computer untouched while it runs.

Example:
  python scripts/idle_baseline.py            # 60 s
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crisismmd import experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--part", default="A", help="coursework part to file the record under")
    parser.add_argument("--member", default=os.environ.get("CRISISMMD_MEMBER"))
    args = parser.parse_args()
    if not args.member:
        parser.error("set --member or export CRISISMMD_MEMBER (e.g. SHC)")
    experiment.enter_repo_root()
    from scripts.run_logger import RunLogger, new_run_config  # group logger, unchanged

    config = new_run_config("templates/idle_baseline_config.json", f"{args.member}_{args.part}_idle_baseline",
                            coursework_part=args.part, baseline_seconds=args.seconds)
    print(f"Measuring idle draw for {args.seconds} s: please don't use the computer...", flush=True)
    with RunLogger(config, root=experiment.LOG_ROOT) as run:
        time.sleep(run.config["baseline_seconds"])
    print(run.describe())


if __name__ == "__main__":
    main()
