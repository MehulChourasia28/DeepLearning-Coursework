#!/usr/bin/env python3
"""Train and evaluate one model on one task. Every run is registered in results/ (see crisismmd/tracking.py).

Models: majority (baseline), text, image, fusion (concatenation). Text encoders: textcnn, bilstm.
The best epoch is chosen by validation macro-F1 with early stopping. The test split is only evaluated
for --run-type final.

Examples:
  python scripts/train.py --task informative --model fusion --run-type debug --fraction 0.1 --max-epochs 2
  python scripts/train.py --task informative --model text --text-encoder bilstm --seed 1 --run-type final

Set your member prefix once per shell (used in experiment IDs, e.g. HEL-A-001):
  export CRISISMMD_MEMBER=HEL
"""
import argparse
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (GPU runs)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

from crisismmd import data, tracking  # noqa: E402
from crisismmd.engine import class_weights, fit, predict  # noqa: E402
from crisismmd.metrics import compute_metrics  # noqa: E402
from crisismmd.models import build_model, count_trainable  # noqa: E402
from crisismmd.text import Vocab, tokenize  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(data.TASK_LABELS))
    p.add_argument("--model", required=True, choices=["majority", "text", "image", "fusion"])
    p.add_argument("--text-encoder", default="textcnn", choices=["textcnn", "bilstm"])
    p.add_argument("--imbalance", default="none", choices=["none", "weighted_loss", "weighted_sampler"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-type", required=True, choices=["debug", "exploratory", "final"])
    p.add_argument("--purpose", default="", help="short free-text reason for this run")
    p.add_argument("--rerun-reason", default="", help="if repeating a run: error, planned replication, ...")
    p.add_argument("--member", default=os.environ.get("CRISISMMD_MEMBER"),
                   help="member prefix for experiment IDs (default: $CRISISMMD_MEMBER)")
    p.add_argument("--part", default=None, help="override coursework part (default A/B from task)")
    # Data
    p.add_argument("--data-root", type=Path, default=data.DATA_ROOT)
    p.add_argument("--features", type=Path, default=data.DEFAULT_FEATURES)
    p.add_argument("--results-dir", type=Path, default=tracking.RESULTS_DIR)
    p.add_argument("--fraction", type=float, default=1.0, help="random fraction of training rows (debug)")
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--min-freq", type=int, default=2, help="minimum training-set count for a vocab token")
    # Model and optimisation
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    # Hardware and tracking
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    p.add_argument("--country", default="GBR", help="ISO code for codecarbon carbon intensity")
    p.add_argument("--no-codecarbon", action="store_true")
    args = p.parse_args()
    if not args.member:
        p.error("set --member or export CRISISMMD_MEMBER (e.g. HEL)")
    if not 0 < args.fraction <= 1:
        p.error("--fraction must be in (0, 1]")
    if args.model in ("majority", "image"):
        args.text_encoder = None
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    part = args.part or {"informative": "A", "humanitarian": "B"}[args.task]
    exp_id = tracking.next_experiment_id(args.member, part, args.results_dir)
    run_dir = args.results_dir / "runs" / exp_id
    run_dir.mkdir(parents=True)
    log_file = (run_dir / "train.log").open("w", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")

    eval_splits = ["valid", "test"] if args.run_type == "final" else ["valid"]
    record = {
        "experiment_id": exp_id, "member": args.member, "part": part, "task": args.task,
        "run_type": args.run_type, "status": "running", "purpose": args.purpose,
        "rerun_reason": args.rerun_reason, "model": args.model, "text_encoder": args.text_encoder,
        "imbalance": args.imbalance, "seed": args.seed, "dataset_fraction": args.fraction,
        "batch_size": args.batch_size, "max_epochs": args.max_epochs, "patience": args.patience,
        "lr": args.lr, **tracking.hardware_info(args.device), "torch_threads": torch.get_num_threads(),
        "image_feature_cache": f"reused: {args.features.name}", "checkpoint_reused": False,
        "evaluated_splits": eval_splits, "started_at": tracking.now(),
        "config": {k: tracking.portable(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    tracking.save_run(run_dir, record, args.results_dir)  # registered before any work starts
    log(f"{exp_id}: task={args.task} model={args.model} text={args.text_encoder} "
        f"imbalance={args.imbalance} seed={args.seed} run_type={args.run_type} device={args.device}")

    started = time.perf_counter()
    train_energy, eval_energy = {}, {}
    try:
        set_seed(args.seed)
        labels = data.TASK_LABELS[args.task]
        num_classes = len(labels)
        image_index, features = data.load_image_features(args.features)
        frames, excluded = {}, 0
        for split in ["train"] + eval_splits:
            frames[split], missing = data.drop_missing_images(data.load_split(args.task, split, args.data_root),
                                                              image_index)
            excluded += missing
        if args.fraction < 1:
            rng = np.random.default_rng(args.seed)
            size = max(1, round(args.fraction * len(frames["train"])))
            keep = np.sort(rng.choice(len(frames["train"]), size=size, replace=False))
            frames["train"] = frames["train"].iloc[keep].reset_index(drop=True)

        uses_text = args.model in ("text", "fusion")
        vocab = (Vocab.build((tokenize(t) for t in frames["train"]["tweet_text"]), min_freq=args.min_freq)
                 if uses_text else None)  # training split only
        datasets = {split: data.encode_frame(frames[split], args.task, vocab, image_index, features,
                                             args.max_len) for split in frames}
        train_labels = datasets["train"].tensors[3].numpy()
        record.update(train_examples=len(train_labels), excluded_rows=excluded,
                      vocab_size=len(vocab) if vocab else None)
        log(f"train rows {len(train_labels):,}, excluded (missing image) {excluded}, "
            f"vocab {len(vocab) if vocab else '-'}")

        # ---- training phase -------------------------------------------------------------------------
        tracker = tracking.EnergyTracker(f"{exp_id}-train", args.country, not args.no_codecarbon)
        record["tracker"] = f"codecarbon {tracker.version}" if tracker.version else "none"
        record["energy_method"] = tracking.energy_method(args.device) if tracker.version else "not measured"
        model, majority, history, best_epoch, stop_reason, params = None, None, [], None, "no_training", 0
        train_started = time.perf_counter()
        tracker.start()
        try:
            if args.model == "majority":
                majority = int(np.bincount(train_labels, minlength=num_classes).argmax())
                log(f"majority class: {labels[majority]}")
            else:
                device = torch.device(args.device)
                model = build_model(args.model, num_classes, len(vocab) if vocab else None,
                                    args.text_encoder or "textcnn", args.emb_dim, args.hidden,
                                    args.dropout).to(device)
                params = count_trainable(model)
                log(f"trainable parameters: {params:,}")
                weights = (class_weights(train_labels, num_classes).to(device)
                           if args.imbalance == "weighted_loss" else None)
                if weights is not None:
                    record["class_weights"] = dict(zip(labels, weights.tolist()))
                criterion = nn.CrossEntropyLoss(weight=weights)
                generator = torch.Generator().manual_seed(args.seed)
                if args.imbalance == "weighted_sampler":
                    counts = np.maximum(np.bincount(train_labels, minlength=num_classes), 1)
                    sampler = WeightedRandomSampler(torch.as_tensor(1.0 / counts[train_labels], dtype=torch.double),
                                                    len(train_labels), replacement=True, generator=generator)
                    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, sampler=sampler)
                else:
                    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                                              generator=generator)
                val_loader = DataLoader(datasets["valid"], batch_size=256)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                best_state, history, best_epoch, stop_reason = fit(
                    model, train_loader, val_loader, criterion, optimizer, device, args.max_epochs,
                    args.patience, log)
                model.load_state_dict(best_state)
        finally:
            train_energy = tracker.stop()
            record["train_time_s"] = round(time.perf_counter() - train_started, 2)
        record.update(trainable_params=params, epochs_completed=len(history), best_epoch=best_epoch,
                      stop_reason=stop_reason)
        if history:
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if model is not None:
            torch.save({"experiment_id": exp_id, "state_dict": model.state_dict(), "labels": labels,
                        "vocab": vocab.itos if vocab else None, "config": record["config"]},
                       run_dir / "model.pt")

        # ---- evaluation phase -----------------------------------------------------------------------
        tracker = tracking.EnergyTracker(f"{exp_id}-eval", args.country, not args.no_codecarbon)
        eval_started = time.perf_counter()
        tracker.start()
        try:
            metrics = {}
            for split in eval_splits:
                truth = datasets[split].tensors[3].numpy()
                if model is None:
                    probs = np.zeros((len(truth), num_classes), dtype=np.float32)
                    probs[:, majority] = 1.0
                else:
                    probs = predict(model, DataLoader(datasets[split], batch_size=256), torch.device(args.device))
                pred = probs.argmax(1)
                metrics[split] = compute_metrics(truth, pred, labels, data.POSITIVE_LABEL.get(args.task))
                out = frames[split][["event_name", "tweet_id", "image_id", "image", "tweet_text", "label"]].copy()
                out.insert(0, "split", split)
                out["pred"] = [labels[i] for i in pred]
                out["correct"] = out["label"] == out["pred"]
                for i, label in enumerate(labels):
                    out[f"prob_{label}"] = probs[:, i]
                out.to_csv(run_dir / f"predictions_{split}.csv", index=False)
        finally:
            eval_energy = tracker.stop()
            record["eval_time_s"] = round(time.perf_counter() - eval_started, 2)
        if majority is not None:
            metrics["majority_class"] = labels[majority]
        tracking.write_json(run_dir / "metrics.json", metrics)

        record.update(status="early-stopped" if stop_reason == "early_stopped" else "completed",
                      val_macro_f1=metrics["valid"]["macro_f1"])
        if "test" in metrics:
            record.update(test_macro_f1=metrics["test"]["macro_f1"], test_accuracy=metrics["test"]["accuracy"])
        log(f"val macro-F1 {metrics['valid']['macro_f1']:.4f}  val acc {metrics['valid']['accuracy']:.4f}"
            + (f"  | test macro-F1 {metrics['test']['macro_f1']:.4f}  test acc {metrics['test']['accuracy']:.4f}"
               if "test" in metrics else ""))
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        raise
    except Exception as error:
        record.update(status="failed", intended_run_type=args.run_type, run_type="failed", error=repr(error))
        log(traceback.format_exc())
        raise
    finally:
        record.update(tracking.summarise_energy(train_energy, eval_energy, args.country))
        record.update(runtime_s=round(time.perf_counter() - started, 2), finished_at=tracking.now())
        tracking.save_run(run_dir, record, args.results_dir)
        energy = record.get("energy_kwh")
        log(f"{exp_id} {record['status']}: runtime {record['runtime_s']}s"
            + (f", energy {energy:.2e} kWh, {record['emissions_kgco2e']:.2e} kgCO2e" if energy else "")
            + f" -> {tracking.portable(run_dir)}")
        log_file.close()


if __name__ == "__main__":
    main()
