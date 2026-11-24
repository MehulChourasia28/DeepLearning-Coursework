"""Full Part A run on CPU, in order. Every step is its own logged run.

    python -m part_a.run_all              # everything
    python -m part_a.run_all --skip-debug # skip the 5% smoke tests

Steps
  1. idle baseline (60 s)                         development / idle_baseline
  2. ResNet18 feature cache (skipped if present)  artefact_production
  3. debug smoke tests: both fusion models, 5% of train, 1 epoch      development
  4. majority-class baseline (deterministic, 1 run) + test evaluation
  5. text_textcnn, text_bilstm, image_mlp, fusion_textcnn, fusion_bilstm,
     seeds 0/1/2, each followed by its test evaluation                final_training / evaluation
  6. tables + figures                                                 artefact_production
"""
import argparse
import subprocess
import sys

from . import config as C


def sh(*args):
    cmd = [sys.executable, "-m", *args]
    print("\n$ python -m " + " ".join(args), flush=True)
    subprocess.run(cmd, check=True, cwd=C.ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-debug", action="store_true")
    ap.add_argument("--skip-idle", action="store_true")
    a = ap.parse_args()
    if not a.skip_idle:
        sh("part_a.idle_baseline", "--seconds", "60")
    if not C.FEATURE_CACHE.exists():
        sh("part_a.extract_features")
    if not a.skip_debug:
        for m in ("fusion_textcnn", "fusion_bilstm"):
            sh("part_a.train", "--model", m, "--seeds", "0", "--run-type", "debug",
               "--fraction", "0.05", "--max-epochs", "1")
    sh("part_a.train", "--model", "majority", "--seeds", "0", "--run-type", "final", "--test")
    for m in C.MODELS:
        sh("part_a.train", "--model", m, "--seeds", *map(str, C.SEEDS), "--run-type", "final",
           "--test")
    sh("part_a.report")


if __name__ == "__main__":
    main()
