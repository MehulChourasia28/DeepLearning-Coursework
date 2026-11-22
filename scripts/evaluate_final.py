#!/usr/bin/env python3
"""Select median-validation runs, then evaluate every final checkpoint on the TEST split.

Order matters (coursework rule: choose confusion-matrix runs before inspecting test results):
  1. Collect the task's completed final training records from logs/runs/.
  2. Write the median-validation selection per model, with a timestamp, BEFORE any test evaluation.
  3. For each final training run without one, create a separate evaluation record (run logger,
     templates/evaluation_config.json) linked by evaluated_experiment_id; it reloads best.pt, predicts the
     test split and saves predictions_test.csv and metrics.json.

Example:
  python scripts/evaluate_final.py --task humanitarian
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from crisismmd import data, experiment  # noqa: E402
from crisismmd.engine import predict  # noqa: E402
from crisismmd.metrics import compute_metrics  # noqa: E402
from crisismmd.models import build_model, count_trainable  # noqa: E402
from crisismmd.text import Vocab  # noqa: E402
from crisismmd.tracking import write_json  # noqa: E402

DONE = {"completed", "early-stopped"}


def load_records(root):
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob("*/summary.json"))]


def median_run(runs):
    ordered = sorted(runs, key=lambda r: (r["validation_macro_f1"], r.get("seed") or 0))
    return ordered[(len(ordered) - 1) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=list(data.TASK_LABELS))
    parser.add_argument("--part", default=None, help="default A/B from task")
    parser.add_argument("--member", default=None, help="only this member's runs (experiment-ID prefix)")
    parser.add_argument("--data-root", type=Path, default=data.DATA_ROOT)
    parser.add_argument("--features", type=Path, default=data.DEFAULT_FEATURES)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.data_root, args.features = args.data_root.resolve(), args.features.resolve()
    experiment.enter_repo_root()
    from scripts.run_logger import RunLogger, new_run_config  # group logger, unchanged

    part = args.part or {"informative": "A", "humanitarian": "B"}[args.task]
    task_name = experiment.TASK_NAMES[args.task]
    records = load_records(experiment.LOG_ROOT)
    training = [r for r in records if r.get("activity") == "training" and r.get("run_type") == "final"
                and r.get("task") == task_name and r.get("coursework_part") == part
                and r.get("run_status") in DONE and r.get("checkpoint_path")
                and (not args.member or r["experiment_id"].startswith(f"{args.member}_"))]
    if not training:
        raise SystemExit(f"No completed final training records for part {part} / {task_name} in {experiment.LOG_ROOT}")
    evaluated = {r.get("evaluated_experiment_id") for r in records
                 if r.get("activity") == "evaluation" and r.get("run_status") == "completed"}

    # 1-2. Median-validation selection, saved before any test evaluation in this call.
    groups = {}
    for r in training:
        groups.setdefault(r["model"], []).append(r)
    already_tested = sorted(r["experiment_id"] for r in training if r["experiment_id"] in evaluated)
    selection = {
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "rule": "median validation macro-F1 across seeds (lower middle for even counts; majority has one run)",
        "test_results_inspected_before_selection": bool(already_tested),
        "runs_already_test_evaluated": already_tested,
        "runs": {key: {"experiment_id": (m := median_run(g))["experiment_id"], "seed": m.get("seed"),
                       "validation_macro_f1": m["validation_macro_f1"],
                       "all_seeds": {str(r.get("seed")): r["validation_macro_f1"] for r in g}}
                 for key, g in sorted(groups.items())},
    }
    selection_dir = Path("results/selection")
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_path = selection_dir / f"part{part}_{args.task}_median_validation_selection.json"
    if selection_path.exists():
        previous = json.loads(selection_path.read_text())
        if previous["runs"] != selection["runs"]:
            stamped = selection_dir / f"part{part}_{args.task}_median_validation_selection_{previous['selected_at'][:19].replace(':', '')}.json"
            selection_path.rename(stamped)
            print(f"selection changed; previous kept as {stamped}")
        else:
            selection = previous  # unchanged: keep the original timestamp
    write_json(selection_path, selection)
    print(f"median-validation selection -> {selection_path}")
    for key, chosen in selection["runs"].items():
        print(f"  {key:32s} {chosen['experiment_id']} (val macro-F1 {chosen['validation_macro_f1']:.4f})")

    # 3. Separate, linked evaluation records.
    image_index, features = data.load_image_features(args.features)
    labels = data.TASK_LABELS[args.task]
    todo = [r for r in training if r["experiment_id"] not in evaluated]
    print(f"{len(todo)} final runs to evaluate on test ({len(training) - len(todo)} already done)")
    for r in todo:
        hp = r.get("hyperparameters") or {}
        uses_image = bool(r.get("cache_used"))
        member = r["experiment_id"].split("_", 1)[0]
        seed_tag = "" if r.get("seed") is None else f"_seed{r['seed']}"
        config = new_run_config(
            "templates/evaluation_config.json", f"{member}_{part}_{r['model']}{seed_tag}_test_eval",
            coursework_part=part, task=task_name, model=r["model"], seed=r.get("seed"),
            training_examples=r.get("training_examples"), dataset_fraction=r.get("dataset_fraction"),
            batch_size=256, hyperparameters={"model_key": r["model"], "vocab_size": hp.get("vocab_size")},
            evaluated_experiment_id=r["experiment_id"], checkpoint_input_path=r["checkpoint_path"],
            cache_used=uses_image, cache_path=r.get("cache_path"),
            measurement_boundary=experiment.EVAL_BOUNDARY,
            reuse_notes="Reuses the selected checkpoint of the linked training run"
                        + (" and cached image features." if uses_image else "; no image features used."),
            notes="Test evaluation after the median-validation selection was recorded.",
        )
        with RunLogger(config, root=experiment.LOG_ROOT) as run:
            checkpoint = torch.load(r["checkpoint_path"], map_location="cpu", weights_only=True)
            cfg = checkpoint["config"]
            frame, missing = data.drop_missing_images(data.load_split(args.task, "test", args.data_root),
                                                      image_index)
            vocab = Vocab(checkpoint["vocab"]) if checkpoint["vocab"] else None
            dataset = data.encode_frame(frame, args.task, vocab, image_index, features, cfg["max_len"])
            truth = dataset.tensors[3].numpy()
            if "majority_class" in checkpoint:
                probs = np.zeros((len(truth), len(labels)), dtype=np.float32)
                probs[:, labels.index(checkpoint["majority_class"])] = 1.0
                run.update(trainable_parameters=0)
            else:
                model = build_model(cfg["model"], len(labels), len(vocab) if vocab else None,
                                    cfg["text_encoder"] or "textcnn", cfg["emb_dim"], cfg["hidden"],
                                    cfg["dropout"]).to(args.device)
                model.load_state_dict(checkpoint["state_dict"])
                probs = predict(model, DataLoader(dataset, batch_size=256), torch.device(args.device))
                run.update(trainable_parameters=count_trainable(model))
            metrics = {"test": compute_metrics(truth, probs.argmax(1), labels, data.POSITIVE_LABEL.get(args.task)),
                       "excluded_rows": missing}
            out = frame[["event_name", "tweet_id", "image_id", "image", "tweet_text", "label"]].copy()
            out.insert(0, "split", "test")
            out["pred"] = [labels[i] for i in probs.argmax(1)]
            out["correct"] = out["label"] == out["pred"]
            for i, label in enumerate(labels):
                out[f"prob_{label}"] = probs[:, i]
            out.to_csv(run.path / "predictions_test.csv", index=False)
            write_json(run.path / "metrics.json", metrics)
            run.progress(split="test", examples=len(truth))
            run.log_test(metrics["test"]["macro_f1"])
            run.update(predictions_path=(run.path / "predictions_test.csv").as_posix())
        print(f"  {r['experiment_id']} -> {run.summary['experiment_id']}: test macro-F1 "
              f"{metrics['test']['macro_f1']:.4f}, accuracy {metrics['test']['accuracy']:.4f}")
    print("Next: python -m scripts.aggregate_runs, then scripts/summarise.py")


if __name__ == "__main__":
    main()
