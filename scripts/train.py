#!/usr/bin/env python3
"""Train one model on one task, logged with the group's run logger (scripts/run_logger.py).

Models: majority (baseline), text, image, fusion (concatenation). Text encoders: textcnn, bilstm.
The best epoch is chosen by validation macro-F1 with early stopping. The TEST split is never used here:
after all final runs, `scripts/evaluate_final.py` records the median-validation selection and then
evaluates each final checkpoint on test in a separate, linked evaluation record.

Every run gets logs/runs/<experiment_id>/ with the logger's config.json, summary.json (energy per
component, uncertainty range, status), epochs.csv and TensorBoard events, plus our own train.log,
history.csv, metrics.json, predictions_valid.csv and best.pt (weights + vocabulary + config).

Examples:
  python scripts/train.py --task humanitarian --model fusion --run-type debug --fraction 0.1 --max-epochs 2
  python scripts/train.py --task humanitarian --model fusion --imbalance weighted_loss --seed 1 --run-type final

Set your member prefix once per shell (first part of experiment IDs):
  export CRISISMMD_MEMBER=SHC
"""
import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (GPU runs)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

from crisismmd import data, experiment  # noqa: E402
from crisismmd.engine import class_weights, fit, predict  # noqa: E402
from crisismmd.metrics import compute_metrics  # noqa: E402
from crisismmd.models import build_model, count_trainable  # noqa: E402
from crisismmd.text import Vocab, tokenize  # noqa: E402
from crisismmd.tracking import write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(data.TASK_LABELS))
    p.add_argument("--model", required=True, choices=["majority", "text", "image", "fusion"])
    p.add_argument("--text-encoder", default="textcnn", choices=["textcnn", "bilstm"])
    p.add_argument("--imbalance", default="none", choices=["none", "weighted_loss", "weighted_sampler"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-type", required=True, choices=["debug", "exploratory", "final"])
    p.add_argument("--purpose", default="", help="short free-text reason for this run (stored as notes)")
    p.add_argument("--rerun-of", default=None, help="experiment_id this run repeats, if any")
    p.add_argument("--rerun-reason", default=None, help="error, planned replication, changed configuration, ...")
    p.add_argument("--member", default=os.environ.get("CRISISMMD_MEMBER"),
                   help="member prefix for experiment IDs (default: $CRISISMMD_MEMBER)")
    p.add_argument("--part", default=None, help="override coursework part (default A/B from task)")
    # Data
    p.add_argument("--data-root", type=Path, default=data.DATA_ROOT)
    p.add_argument("--features", type=Path, default=data.DEFAULT_FEATURES)
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
    p.add_argument("--no-codecarbon", action="store_true", help="disable energy tracking (not for real runs)")
    args = p.parse_args()
    if not args.member:
        p.error("set --member or export CRISISMMD_MEMBER (e.g. SHC)")
    if not 0 < args.fraction <= 1:
        p.error("--fraction must be in (0, 1]")
    if args.model in ("majority", "image"):
        args.text_encoder = None
    if args.model == "majority" and args.imbalance != "none":
        p.error("--imbalance does not apply to the majority baseline")
    args.data_root, args.features = args.data_root.resolve(), args.features.resolve()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def predictions_frame(frame, split, labels, probs):
    out = frame[["event_name", "tweet_id", "image_id", "image", "tweet_text", "label"]].copy()
    out.insert(0, "split", split)
    out["pred"] = [labels[i] for i in probs.argmax(1)]
    out["correct"] = out["label"] == out["pred"]
    for i, label in enumerate(labels):
        out[f"prob_{label}"] = probs[:, i]
    return out


def main():
    args = parse_args()
    experiment.enter_repo_root()
    from scripts.run_logger import RunLogger, new_run_config  # group logger, unchanged

    if args.threads:
        torch.set_num_threads(args.threads)
    part = args.part or {"informative": "A", "humanitarian": "B"}[args.task]
    key = experiment.model_key(args.model, args.text_encoder, args.imbalance)
    majority = args.model == "majority"
    uses_image = args.model in ("image", "fusion")
    seed_tag = "" if majority else f"_seed{args.seed}"
    hyperparameters = {
        "model_key": key, "text_encoder": args.text_encoder, "imbalance": args.imbalance,
        "max_tokens": args.max_len, "min_freq": args.min_freq, "emb_dim": args.emb_dim,
        "head_hidden": args.hidden, "dropout": args.dropout, "weight_decay": args.weight_decay,
        "loss": "cross_entropy" + (" (class-weighted)" if args.imbalance == "weighted_loss" else ""),
        "selection": "best validation macro-F1", "member": args.member,
        "code": "crisismmd package + scripts/train.py",
    }
    config = new_run_config(
        "templates/run_config.json", f"{args.member}_{part}_{key}{seed_tag}_{args.run_type}",
        coursework_part=part, task=experiment.TASK_NAMES[args.task], activity="training",
        workflow_stage=experiment.workflow_stage(part, args.run_type), run_type=args.run_type, model=key,
        seed=None if majority else args.seed, training_examples=None, dataset_fraction=args.fraction,
        batch_size=None if majority else args.batch_size,
        max_epochs=None if majority else args.max_epochs,
        early_stopping_patience=None if majority else args.patience,
        optimizer=None if majority else "Adam", initial_lr=None if majority else args.lr,
        scheduler=None, hyperparameters=hyperparameters, measurement_boundary=experiment.TRAIN_BOUNDARY,
        cache_used=uses_image, cache_path=experiment.relative(args.features) if uses_image else None,
        reuse_notes=("Frozen ResNet18 features reused from the cached .npz; no image decoding or ResNet18 "
                     "inference." if uses_image else "No image features used by this model."),
        rerun_of=args.rerun_of, rerun_reason=args.rerun_reason, notes=args.purpose or None,
    )

    labels = data.TASK_LABELS[args.task]
    num_classes = len(labels)
    log_file = None
    with RunLogger(config, root=experiment.LOG_ROOT, track_energy=not args.no_codecarbon) as run:
        log_file = (run.path / "train.log").open("w", encoding="utf-8")

        def log(message):
            print(message, flush=True)
            log_file.write(message + "\n")

        try:
            log(f"{run.summary['experiment_id']}: task={args.task} model={key} seed="
                f"{'-' if majority else args.seed} run_type={args.run_type} device={args.device}")
            set_seed(args.seed)
            image_index, features = data.load_image_features(args.features)
            frames, excluded = {}, 0
            for split in ("train", "valid"):  # the test split is never loaded during training
                frames[split], missing = data.drop_missing_images(
                    data.load_split(args.task, split, args.data_root), image_index)
                excluded += missing
            if args.fraction < 1:
                rng = np.random.default_rng(args.seed)
                size = max(1, round(args.fraction * len(frames["train"])))
                keep = np.sort(rng.choice(len(frames["train"]), size=size, replace=False))
                frames["train"] = frames["train"].iloc[keep].reset_index(drop=True)
            vocab = (Vocab.build((tokenize(t) for t in frames["train"]["tweet_text"]), min_freq=args.min_freq)
                     if args.model in ("text", "fusion") else None)  # training split only
            datasets = {split: data.encode_frame(frames[split], args.task, vocab, image_index, features,
                                                 args.max_len) for split in frames}
            train_labels = datasets["train"].tensors[3].numpy()
            hyperparameters.update(vocab_size=len(vocab) if vocab else None, excluded_rows=excluded)
            run.update(training_examples=len(train_labels), hyperparameters=hyperparameters)
            log(f"train rows {len(train_labels):,}, excluded (missing image) {excluded}, "
                f"vocab {len(vocab) if vocab else '-'}")

            checkpoint = {"experiment_id": run.summary["experiment_id"], "labels": labels,
                          "vocab": vocab.itos if vocab else None, "config": {
                              "task": args.task, "model": args.model, "text_encoder": args.text_encoder,
                              "imbalance": args.imbalance, "seed": None if majority else args.seed,
                              "emb_dim": args.emb_dim, "hidden": args.hidden, "dropout": args.dropout,
                              "max_len": args.max_len, "min_freq": args.min_freq}}
            if majority:
                majority_id = int(np.bincount(train_labels, minlength=num_classes).argmax())
                run.progress(split="train", examples=len(train_labels))
                probs = np.zeros((len(datasets["valid"]), num_classes), dtype=np.float32)
                probs[:, majority_id] = 1.0
                checkpoint["majority_class"] = labels[majority_id]
                run.update(trainable_parameters=0)
                log(f"majority class: {labels[majority_id]}")
            else:
                device = torch.device(args.device)
                model = build_model(args.model, num_classes, len(vocab) if vocab else None,
                                    args.text_encoder or "textcnn", args.emb_dim, args.hidden,
                                    args.dropout).to(device)
                run.update(trainable_parameters=count_trainable(model))
                log(f"trainable parameters: {count_trainable(model):,}")
                weights = (class_weights(train_labels, num_classes).to(device)
                           if args.imbalance == "weighted_loss" else None)
                if weights is not None:
                    hyperparameters["class_weights"] = dict(zip(labels, [round(w, 4) for w in weights.tolist()]))
                    run.update(hyperparameters=hyperparameters)
                    log("class weights: " + ", ".join(f"{k} {v}" for k, v in hyperparameters["class_weights"].items()))
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
                    args.patience, log,
                    on_batch=lambda epoch, batch, loss, n: run.log_batch(loss, optimizer, epoch=epoch,
                                                                         batch=batch, examples=n),
                    on_epoch=lambda epoch, row: run.log_epoch(
                        epoch, train_loss=row["train_loss"], valid_loss=row["val_loss"],
                        valid_macro_f1=row["val_macro_f1"], optimizer=optimizer))
                if stop_reason == "early_stopped":
                    run.early_stop()
                model.load_state_dict(best_state)
                checkpoint["state_dict"] = model.state_dict()
                pd.DataFrame(history).to_csv(run.path / "history.csv", index=False)
                probs = predict(model, val_loader, device)
                log(f"selected epoch {best_epoch} ({stop_reason})")

            torch.save(checkpoint, run.path / "best.pt")
            metrics = {"valid": compute_metrics(datasets["valid"].tensors[3].numpy(), probs.argmax(1), labels,
                                                data.POSITIVE_LABEL.get(args.task))}
            write_json(run.path / "metrics.json", metrics)
            predictions_frame(frames["valid"], "valid", labels, probs).to_csv(
                run.path / "predictions_valid.csv", index=False)
            run.update(validation_macro_f1=metrics["valid"]["macro_f1"],
                       checkpoint_path=(run.path / "best.pt").as_posix(),
                       predictions_path=(run.path / "predictions_valid.csv").as_posix())
            log(f"val macro-F1 {metrics['valid']['macro_f1']:.4f}  val acc {metrics['valid']['accuracy']:.4f}")
        finally:
            log_file.close()
    print(run.describe())
    if args.run_type == "final":
        print("Test not evaluated here: after all final runs, use scripts/evaluate_final.py")


if __name__ == "__main__":
    main()
