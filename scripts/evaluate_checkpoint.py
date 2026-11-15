#!/usr/bin/env python3
"""Rebuild a saved model from its checkpoint and evaluate it on one split.

The checkpoint holds the weights, the training vocabulary, the class order and the training config, so
no retraining is needed. Example:
  python scripts/evaluate_checkpoint.py checkpoints/part_a_informative_fusion_textcnn_SHC-A-026.pt --split test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from crisismmd import data  # noqa: E402
from crisismmd.engine import predict  # noqa: E402
from crisismmd.metrics import compute_metrics  # noqa: E402
from crisismmd.models import build_model  # noqa: E402
from crisismmd.text import Vocab  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=list(data.SPLITS))
    parser.add_argument("--data-root", type=Path, default=data.DATA_ROOT)
    parser.add_argument("--features", type=Path, default=data.DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=None, help="optional CSV of per-example predictions")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config, labels = checkpoint["config"], checkpoint["labels"]
    vocab = Vocab(checkpoint["vocab"]) if checkpoint["vocab"] else None
    model = build_model(config["model"], len(labels), len(vocab) if vocab else None,
                        config["text_encoder"] or "textcnn", config["emb_dim"], config["hidden"], config["dropout"])
    model.load_state_dict(checkpoint["state_dict"])

    image_index, features = data.load_image_features(args.features)
    frame, missing = data.drop_missing_images(data.load_split(config["task"], args.split, args.data_root),
                                              image_index)
    dataset = data.encode_frame(frame, config["task"], vocab, image_index, features, config["max_len"])
    probs = predict(model, DataLoader(dataset, batch_size=256), torch.device("cpu"))
    metrics = compute_metrics(dataset.tensors[3].numpy(), probs.argmax(1), labels,
                              data.POSITIVE_LABEL.get(config["task"]))

    print(f"{checkpoint['experiment_id']}: {config['model']} ({config['text_encoder']}) on "
          f"{config['task']}/{args.split}, {len(frame):,} rows ({missing} excluded)")
    print(f"macro-F1 {metrics['macro_f1']:.4f}  accuracy {metrics['accuracy']:.4f}  "
          f"weighted-F1 {metrics['weighted_f1']:.4f}")
    for label, scores in metrics["per_class"].items():
        print(f"  {label:40s} P {scores['precision']:.3f}  R {scores['recall']:.3f}  F1 {scores['f1']:.3f}  "
              f"correct {scores['correct']}/{scores['support']}")
    if args.out:
        out = frame[["tweet_id", "image_id", "label"]].copy()
        out["pred"] = [labels[i] for i in probs.argmax(1)]
        for i, label in enumerate(labels):
            out[f"prob_{label}"] = probs[:, i]
        out.to_csv(args.out, index=False)
        print(f"predictions -> {args.out}")


if __name__ == "__main__":
    main()
