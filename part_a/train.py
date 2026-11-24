"""Train / evaluate Part A models with one energy record per run.

Examples
    # quick debug check on 5% of training data, 1 epoch (logged as run_type=debug)
    python -m part_a.train --model fusion_textcnn --seeds 0 --run-type debug --fraction 0.05 --max-epochs 1

    # final runs: trains seeds 0,1,2 then evaluates each selected checkpoint on test
    python -m part_a.train --model fusion_textcnn --seeds 0 1 2 --run-type final --test

Models: majority, text_textcnn, text_bilstm, image_mlp, fusion_textcnn, fusion_bilstm
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config as C
from .data import build_tensors
from .energy import RunLogger
from .metrics import compute
from .models import Classifier, trainable_parameters

BOUNDARY_TRAIN = (
    "Whole training process: loading the cached image features and TSVs, tokenisation, "
    "training-only vocabulary, model construction, all epochs with validation after each, "
    "saving the best checkpoint, history and validation predictions. Excludes test "
    "evaluation (separate record) and the ResNet18 feature extraction (separate record).")
BOUNDARY_EVAL = (
    "Loading data and the selected checkpoint, predicting the test split, computing metrics "
    "and saving per-example predictions/probabilities.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def predict(model, split, batch_size=512):
    model.eval()
    probs = []
    for i in range(0, len(split["y"]), batch_size):
        sl = slice(i, i + batch_size)
        logits = model(split["ids"][sl], split["lengths"][sl], split["img"][sl])
        probs.append(torch.softmax(logits, 1))
    return torch.cat(probs)


def save_predictions(split, probs, path):
    df = split["df"]
    pred = probs.argmax(1).numpy()
    out = pd.DataFrame({
        "tweet_id": df["tweet_id"], "image_id": df["image_id"], "event_name": df["event_name"],
        "label": df["label"], "pred": [C.LABELS[i] for i in pred],
        "prob_informative": probs[:, 0].numpy().round(6),
        "prob_not_informative": probs[:, 1].numpy().round(6),
        "correct": (pred == split["y"].numpy()), "image": df["image"],
        "tweet_text": df["tweet_text"]})
    out.to_csv(path, index=False, encoding="utf-8")
    return pred


# ---------------------------------------------------------------------------
def train_one(model_kind, seed, run_type, fraction, max_epochs, rerun_of="", rerun_reason=""):
    hp = dict(C.HPARAMS, max_epochs=max_epochs)
    stage = "final_training" if run_type == "final" else "development"
    n_train = round(9601 * fraction)
    with RunLogger(
        run_type=run_type, workflow_stage=stage, activity="training", model=model_kind,
        seed=seed, training_examples=n_train, dataset_fraction=fraction,
        batch_size=hp["batch_size"], measurement_boundary=BOUNDARY_TRAIN,
        cache_used=model_kind != "majority" and "text_" not in model_kind,
        cache_path=C.FEATURE_CACHE.relative_to(C.ROOT).as_posix(),
        reuse_notes=("Frozen ResNet18 features read from the cache; no image decoding or CNN "
                     "inference during training." if "text_" not in model_kind else
                     "Text-only model: image cache loaded but not used by the model."),
        rerun_of=rerun_of, rerun_reason=rerun_reason, hyperparameters=hp,
    ) as run:
        set_seed(seed)
        vocab, data = build_tensors(fraction, seed)
        tr, va = data["train"], data["valid"]
        run.update(training_examples=len(tr["y"]))

        if model_kind == "majority":
            run.mark_training_start()
            majority = int(torch.mode(tr["y"]).values)
            probs = torch.zeros(len(va["y"]), 2)
            probs[:, majority] = 1.0
            pred = save_predictions(va, probs, run.path / "predictions_valid.csv")
            run.mark_training_end()
            m = compute(va["y"].numpy(), pred)
            torch.save({"kind": "majority", "majority_class": majority}, run.path / "best.pt")
            run.update(validation_macro_f1=m["macro_f1"], epochs_completed=0,
                       trainable_parameters=0, examples_processed=len(tr["y"]),
                       checkpoint_path=(run.path / "best.pt").relative_to(C.ROOT).as_posix(),
                       predictions_path=(run.path / "predictions_valid.csv").relative_to(C.ROOT).as_posix(),
                       metrics={"valid": m}, stopping_point="n/a (counts training labels)")
            return run.id, run.path / "best.pt"

        model = Classifier(model_kind, len(vocab), hp)
        n_params = trainable_parameters(model)
        run.update(trainable_parameters=n_params)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
        loss_fn = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(seed)

        best_f1, best_epoch, bad, history, processed = -1.0, 0, 0, [], 0
        run.mark_training_start()
        for epoch in range(1, max_epochs + 1):
            t0 = time.perf_counter()
            model.train()
            perm = torch.randperm(len(tr["y"]), generator=g)
            total = 0.0
            for i in range(0, len(perm), hp["batch_size"]):
                idx = perm[i:i + hp["batch_size"]]
                opt.zero_grad()
                loss = loss_fn(model(tr["ids"][idx], tr["lengths"][idx], tr["img"][idx]),
                               tr["y"][idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
                processed += len(idx)
            train_loss = total / len(perm)
            probs = predict(model, va)
            val_loss = nn.functional.nll_loss(torch.log(probs + 1e-12), va["y"]).item()
            vm = compute(va["y"].numpy(), probs.argmax(1).numpy())
            history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": val_loss,
                            "valid_macro_f1": vm["macro_f1"], "valid_accuracy": vm["accuracy"],
                            "epoch_seconds": round(time.perf_counter() - t0, 3)})
            print(f"  [{run.id}] {model_kind} seed{seed} epoch {epoch:2d} "
                  f"loss {train_loss:.4f} val_loss {val_loss:.4f} val_F1 {vm['macro_f1']:.4f}",
                  flush=True)
            if vm["macro_f1"] > best_f1:
                best_f1, best_epoch, bad = vm["macro_f1"], epoch, 0
                torch.save({"kind": model_kind, "state_dict": model.state_dict(), "hp": hp,
                            "vocab": vocab.itos, "epoch": epoch, "seed": seed},
                           run.path / "best.pt")
                best_probs = probs
            else:
                bad += 1
            run.update(epochs_completed=epoch, selected_epoch=best_epoch,
                       examples_processed=processed, validation_macro_f1=best_f1)
            if bad >= hp["patience"]:
                run.early_stop(f"epoch {epoch} (no val macro-F1 gain for {hp['patience']} epochs)")
                break
        run.mark_training_end()
        if not run.record["stopping_point"]:
            run.update(stopping_point=f"max_epochs={max_epochs} reached")

        pd.DataFrame(history).to_csv(run.path / "history.csv", index=False)
        pred = save_predictions(va, best_probs, run.path / "predictions_valid.csv")
        run.update(
            metrics={"valid": compute(va["y"].numpy(), pred)},
            training_seconds=round(sum(h["epoch_seconds"] for h in history), 2),
            checkpoint_path=(run.path / "best.pt").relative_to(C.ROOT).as_posix(),
            predictions_path=(run.path / "predictions_valid.csv").relative_to(C.ROOT).as_posix())
        return run.id, run.path / "best.pt"


def evaluate_test(model_kind, seed, train_id, checkpoint):
    with RunLogger(
        run_type="final", workflow_stage="evaluation", activity="test_evaluation",
        model=model_kind, seed=seed, training_examples=9601, dataset_fraction=1.0,
        batch_size=512, measurement_boundary=BOUNDARY_EVAL, cache_used=True,
        cache_path=C.FEATURE_CACHE.relative_to(C.ROOT).as_posix(), checkpoint_used=True,
        checkpoint_input_path=checkpoint.relative_to(C.ROOT).as_posix(),
        evaluated_experiment_id=train_id,
        reuse_notes="Selected checkpoint reloaded; no retraining for test evaluation.",
    ) as run:
        ckpt = torch.load(checkpoint, weights_only=False)
        vocab, data = build_tensors(1.0)
        te = data["test"]
        run.mark_training_start()
        if ckpt["kind"] == "majority":
            probs = torch.zeros(len(te["y"]), 2)
            probs[:, ckpt["majority_class"]] = 1.0
            n_params = 0
        else:
            assert ckpt["vocab"] == vocab.itos, "vocabulary differs from training run"
            model = Classifier(model_kind, len(vocab), ckpt["hp"])
            model.load_state_dict(ckpt["state_dict"])
            n_params = trainable_parameters(model)
            probs = predict(model, te)
        pred = save_predictions(te, probs, run.path / "predictions_test.csv")
        run.mark_training_end()
        m = compute(te["y"].numpy(), pred)
        run.update(test_macro_f1=m["macro_f1"], metrics={"test": m}, examples_processed=len(pred),
                   trainable_parameters=n_params,
                   predictions_path=(run.path / "predictions_test.csv").relative_to(C.ROOT).as_posix())
        print(f"  [{run.id}] TEST {model_kind} seed{seed}: macro-F1 {m['macro_f1']:.4f} "
              f"acc {m['accuracy']:.4f}", flush=True)
        return run.id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["majority", *C.MODELS])
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--run-type", default="final", choices=["debug", "exploratory", "final"])
    ap.add_argument("--fraction", type=float, default=1.0, help="training-data fraction")
    ap.add_argument("--max-epochs", type=int, default=C.HPARAMS["max_epochs"])
    ap.add_argument("--test", action="store_true", help="evaluate on test (final runs only)")
    ap.add_argument("--rerun-of", default="", help="experiment_id this run repeats")
    ap.add_argument("--rerun-reason", default="")
    a = ap.parse_args()
    if a.test and (a.run_type != "final" or a.fraction != 1.0):
        ap.error("--test is only allowed for final runs on the full training set")
    for seed in a.seeds:
        train_id, ckpt = train_one(a.model, seed, a.run_type, a.fraction, a.max_epochs,
                                   a.rerun_of, a.rerun_reason)
        if a.test:
            evaluate_test(a.model, seed, train_id, ckpt)


if __name__ == "__main__":
    main()
