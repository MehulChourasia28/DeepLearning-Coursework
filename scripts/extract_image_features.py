#!/usr/bin/env python3
"""Cache frozen ResNet18 (ImageNet) 512-d image features for every image used by the prepared splits.

The encoder is torchvision's ImageNet-pretrained ResNet18 in eval mode with the final fully connected
layer removed, so each image maps to its 512-d global-average-pooled feature. Images are preprocessed
with the transforms prescribed by the weights (resize 256, centre crop 224, ImageNet normalisation).

Output (in --out-dir, default data/crisismmd/features/):
  resnet18_imagenet1k_v1.npz   arrays: image (str paths relative to data/crisismmd/), features (N x 512 float32)
  resnet18_imagenet1k_v1.json  metadata: versions, device, runtime, energy (if codecarbon is installed)

Run once and reuse the cache for every image-only and fusion model.
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crisismmd.tracking import cpu_model  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "crisismmd"
NAME = "resnet18_imagenet1k_v1"


class ImageFiles(Dataset):
    def __init__(self, root, paths, transform):
        self.root, self.paths, self.transform = root, paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root / self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def required_images(root):
    """Unique image paths referenced by any prepared TSV, in sorted order."""
    paths = set()
    for tsv in sorted((root / "prepared").glob("*/*.tsv")):
        with tsv.open(encoding="utf-8", newline="") as stream:
            paths.update(row["image"] for row in csv.DictReader(stream, delimiter="\t"))
    if not paths:
        raise SystemExit(f"No prepared TSVs found under {root / 'prepared'}; run prepare_crisismmd.py first.")
    return sorted(paths)


def start_tracker(enabled, country):
    if not enabled:
        return None
    try:
        from codecarbon import OfflineEmissionsTracker
    except ImportError:
        print("codecarbon not installed: recording runtime only.", flush=True)
        return None
    # Offline mode (no IP geolocation) and no emissions.csv: only the numbers we need are kept.
    tracker = OfflineEmissionsTracker(project_name=NAME, country_iso_code=country,
                                      log_level="error", save_to_file=False)
    tracker.start()
    return tracker


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <data-root>/features")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-codecarbon", action="store_true", help="skip energy tracking")
    parser.add_argument("--country", default="GBR", help="ISO code for codecarbon's grid carbon intensity")
    parser.add_argument("--limit", type=int, default=None, help="debug: only encode the first N images")
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = required_images(args.data_root)
    if args.limit:
        paths = paths[:args.limit]
    print(f"Encoding {len(paths):,} images on {args.device}", flush=True)

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().requires_grad_(False).to(args.device)
    loader = DataLoader(ImageFiles(args.data_root, paths, weights.transforms()), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=args.device.startswith("cuda"))

    tracker = start_tracker(not args.no_codecarbon, args.country)
    started = time.perf_counter()
    chunks = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            chunks.append(model(batch.to(args.device, non_blocking=True)).float().cpu().numpy())
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"  {min(batch_index * args.batch_size, len(paths)):,}/{len(paths):,}", flush=True)
    runtime = time.perf_counter() - started
    emissions = tracker.stop() if tracker else None
    features = np.concatenate(chunks).astype(np.float32)
    assert features.shape == (len(paths), 512), features.shape
    assert np.isfinite(features).all()

    suffix = f"_limit{args.limit}" if args.limit else ""
    np.savez(out_dir / f"{NAME}{suffix}.npz", image=np.array(paths), features=features)
    metadata = {
        "encoder": "torchvision resnet18, ImageNet IMAGENET1K_V1 weights, fc removed, eval mode, frozen",
        "preprocessing": str(weights.transforms()),
        "num_images": len(paths), "feature_dim": int(features.shape[1]),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else cpu_model(),
        "batch_size": args.batch_size, "runtime_seconds": round(runtime, 2),
        "emissions_kgco2e": emissions, "carbon_country": args.country if tracker else None,
        "energy_kwh": tracker.final_emissions_data.energy_consumed if tracker else None,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "python": platform.python_version(),
    }
    (out_dir / f"{NAME}{suffix}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {features.shape} features in {runtime:.1f}s to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
