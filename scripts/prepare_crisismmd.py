#!/usr/bin/env python3
"""Download official coursework splits and their images; audit without changing splits.

Requires requests and Pillow. Re-run to resume; --verify-only avoids network access.
"""
import argparse
import csv
import hashlib
import io
import json
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data" / "crisismmd"
GITHUB_REV = "8862149a36f485f980cfd6ac719cba65e296647c"
HUGGINGFACE_REV = "10a5626ba112ab9c50369c473a2062bcf205d708"
EXPECTED = {"informative": {"train": 9601, "dev": 1573, "test": 1534},
            "humanitarian": {"train": 6126, "dev": 998, "test": 955}}
LABELS = {"informative": {"informative", "not_informative"},
          "humanitarian": {"affected_individuals", "infrastructure_and_utility_damage",
                           "rescue_volunteering_or_donation_effort",
                           "other_relevant_information", "not_humanitarian"}}
MERGE = {"injured_or_dead_people": "affected_individuals",
         "missing_or_found_people": "affected_individuals",
         "vehicle_damage": "infrastructure_and_utility_damage"}


LOCAL = threading.local()
RATE_LOCK = threading.Lock()
RETRY_AT = 0.0


def get(url):
    global RETRY_AT
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(7):
        try:
            while True:
                with RATE_LOCK:
                    remaining = RETRY_AT - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(10, remaining))
            response = LOCAL.session.get(url, timeout=(30, 120))
            if response.status_code == 429:
                delay = max(1, int(response.headers.get("Retry-After", "60"))) + 2
                with RATE_LOCK:
                    if RETRY_AT <= time.monotonic():
                        print(f"Mirror rate limit: waiting {delay}s before resuming.", flush=True)
                    RETRY_AT = max(RETRY_AT, time.monotonic() + delay)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 6:
                raise
            time.sleep(min(60, 2 ** attempt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "splits").mkdir(exist_ok=True)
    metadata_path = ROOT / "source_manifest.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {"github_revision": GITHUB_REV, "huggingface_revision": HUGGINGFACE_REV,
                    "mirror": "https://huggingface.co/datasets/QCRI/CrisisMMD"}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    tables = {}
    for task in EXPECTED:
        tables[task] = {}
        for split in EXPECTED[task]:
            filename = f"task_{task}_text_img_agreed_lab_{split}.tsv"
            path = ROOT / "splits" / filename
            if not args.verify_only and not path.exists():
                path.write_bytes(get(f"https://raw.githubusercontent.com/firojalam/multimodal_social_media/{GITHUB_REV}/data/task_data/{filename}").content)
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            assert len(rows) == EXPECTED[task][split], (task, split, len(rows))
            for row in rows:
                original = row["label"]
                row["label_original"] = original
                row["label"] = MERGE.get(original, original) if task == "humanitarian" else original
            assert set(r["label"] for r in rows) == LABELS[task]
            assert all(r["label_text"] == r["label_image"] == r["label"] for r in rows)
            assert len({(r["tweet_id"], r["image_id"]) for r in rows}) == len(rows)
            tables[task][split] = rows
            prepared = ROOT / "prepared" / task
            prepared.mkdir(parents=True, exist_ok=True)
            with (prepared / ("valid.tsv" if split == "dev" else f"{split}.tsv")).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            print(f"{task}/{split}: {len(rows):,} rows", flush=True)
    paths = sorted({r["image"] for splits in tables.values() for rows in splits.values() for r in rows})
    for p in paths:
        assert p.startswith("data_image/") and ".." not in Path(p).parts
    print(f"Images required: {len(paths):,}", flush=True)
    previous_report = ROOT / "verification_report.json"
    previously_invalid = (json.loads(previous_report.read_text()).get("invalid_images", {})
                          if previous_report.exists() else {})

    def download(path):
        target = ROOT / path
        if target.exists() and path not in previously_invalid:
            return
        content = get(f"https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/{metadata['huggingface_revision']}/{path}").content
        # Ensure an error page cannot be saved as an image.
        with Image.open(io.BytesIO(content)) as im:
            im.verify()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(target)

    if not args.verify_only:
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download, p): p for p in paths}
            for count, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as exc:
                    failures.append({"image": futures[future], "error": str(exc)})
                if count % 250 == 0 or count == len(paths):
                    print(f"Downloaded/present {count:,}/{len(paths):,}; failures {len(failures)}", flush=True)
        (ROOT / "download_failures.json").write_text(json.dumps(failures, indent=2) + "\n")

    invalid = {}
    hashes = {}
    for p in paths:
        try:
            content = (ROOT / p).read_bytes()
            with Image.open(io.BytesIO(content)) as im:
                im.verify()
            with Image.open(io.BytesIO(content)) as im:
                im.convert("RGB").load()
            hashes[p] = hashlib.sha256(content).hexdigest()
        except Exception as exc:
            invalid[p] = str(exc)
    report = {"sources": metadata, "unique_images_required": len(paths),
              "valid_images": len(hashes), "invalid_images": invalid, "tasks": {}}
    for task, splits in tables.items():
        report["tasks"][task] = {}
        for split, rows in splits.items():
            report["tasks"][task][split] = {
                "expected": EXPECTED[task][split], "rows": len(rows),
                "usable_rows": sum(r["image"] not in invalid for r in rows),
                "class_counts": dict(sorted(Counter(r["label"] for r in rows).items())),
                "tsv_sha256": hashlib.sha256((ROOT / "splits" / f"task_{task}_text_img_agreed_lab_{split}.tsv").read_bytes()).hexdigest()}
        overlaps = {}
        for first, second in [("train", "dev"), ("train", "test"), ("dev", "test")]:
            overlaps[f"{first}/{second}"] = {}
            for field in ["tweet_id", "image_id"]:
                shared = sorted({r[field] for r in splits[first]} & {r[field] for r in splits[second]})
                overlaps[f"{first}/{second}"][field] = {"count": len(shared), "ids": shared}
        report["tasks"][task]["overlaps"] = overlaps
    assert report["tasks"]["humanitarian"]["test"]["class_counts"]["affected_individuals"] == 9
    report["passed"] = not invalid
    (ROOT / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "image_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Verified {len(hashes):,}/{len(paths):,} images. Report: {ROOT / 'verification_report.json'}", flush=True)
    if invalid:
        raise SystemExit("Some images are missing or invalid; splits have been preserved. Re-run to retry downloads.")


if __name__ == "__main__":
    main()
