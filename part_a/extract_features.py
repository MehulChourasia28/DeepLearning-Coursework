"""Cache 512-d features from a frozen ImageNet ResNet18 (eval mode, prescribed preprocessing).

    python -m part_a.extract_features              # full cache (logged as artefact_production)
    python -m part_a.extract_features --limit 64   # debug check, writes a separate file
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from . import config as C
from .data import load_split
from .energy import RunLogger


class _Images(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(C.DATA / self.paths[i]) as im:
            return self.transform(im.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="debug: first N images only")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    paths = sorted({p for s in ("train", "valid", "test") for p in load_split(s)["image"]})
    if args.limit:
        paths = paths[:args.limit]
    out = C.FEATURE_CACHE if not args.limit else C.FEATURE_CACHE.with_name(
        f"resnet18_imagenet1k_v1_limit{args.limit}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    weights = ResNet18_Weights.IMAGENET1K_V1
    with RunLogger(
        run_type="debug" if args.limit else "final",
        workflow_stage="development" if args.limit else "artefact_production",
        activity="feature_extraction", model="frozen_resnet18_IMAGENET1K_V1_512d",
        training_examples=9601, dataset_fraction=1.0, batch_size=args.batch_size,
        measurement_boundary=(
            "Loading pretrained ResNet18, decoding every Part A image, prescribed "
            "preprocessing (resize 256, centre-crop 224, ImageNet normalisation), frozen "
            "inference and writing the .npz cache. Excludes dataset download."),
        notes=f"{len(paths)} unique images from the informative train/valid/test splits.",
    ) as run:
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()       # 512-d pooled features
        model.eval().requires_grad_(False)
        loader = DataLoader(_Images(paths, weights.transforms()), batch_size=args.batch_size,
                            num_workers=args.workers)
        run.mark_training_start()
        feats = []
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                feats.append(model(batch).numpy().astype(np.float32))
                if i % 20 == 0:
                    print(f"  {min((i + 1) * args.batch_size, len(paths))}/{len(paths)} images",
                          flush=True)
        run.mark_training_end()
        feats = np.concatenate(feats)
        np.savez(out, paths=np.array(paths), features=feats)
        out.with_suffix(".json").write_text(json.dumps({
            "weights": "IMAGENET1K_V1", "images": len(paths), "dim": int(feats.shape[1]),
            "experiment_id": run.id}, indent=2))
        run.update(examples_processed=len(paths), trainable_parameters=0,
                   cache_output_path=out.relative_to(C.ROOT).as_posix())
    print(f"Saved {feats.shape} to {out}")


if __name__ == "__main__":
    main()
