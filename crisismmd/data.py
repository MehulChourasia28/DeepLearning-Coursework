"""Loading prepared splits, cached image features and building in-memory tensors."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from .text import tokenize

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "crisismmd"
DEFAULT_FEATURES = DATA_ROOT / "features" / "resnet18_imagenet1k_v1.npz"

# Fixed class order per task (index = class id in every output).
TASK_LABELS = {
    "informative": ["not_informative", "informative"],
    "humanitarian": ["affected_individuals", "infrastructure_and_utility_damage",
                     "rescue_volunteering_or_donation_effort", "other_relevant_information",
                     "not_humanitarian"],
}
POSITIVE_LABEL = {"informative": "informative"}
SPLITS = ("train", "valid", "test")


def load_split(task, split, data_root=DATA_ROOT):
    """Prepared TSV as strings (IDs must never become numbers); empty fields stay ''."""
    return pd.read_csv(Path(data_root) / "prepared" / task / f"{split}.tsv", sep="\t", dtype=str,
                       keep_default_na=False)


def load_image_features(path=DEFAULT_FEATURES):
    with np.load(path) as cache:
        paths, features = cache["image"].tolist(), cache["features"].astype(np.float32)
    return {p: i for i, p in enumerate(paths)}, features


def drop_missing_images(frame, image_index):
    """Exclude rows whose image has no cached feature (consistent exclusion across all models)."""
    keep = frame["image"].isin(image_index)
    return frame[keep].reset_index(drop=True), int((~keep).sum())


def encode_frame(frame, task, vocab, image_index, features, max_len):
    """Tensors for one split: token ids, lengths, image features and label ids (row order preserved)."""
    label_to_id = {label: i for i, label in enumerate(TASK_LABELS[task])}
    if vocab is not None:
        encoded = [vocab.encode(tokenize(text), max_len) for text in frame["tweet_text"]]
        tokens = torch.tensor([ids for ids, _ in encoded], dtype=torch.long)
        lengths = torch.tensor([length for _, length in encoded], dtype=torch.long)
    else:
        tokens = torch.zeros((len(frame), 1), dtype=torch.long)
        lengths = torch.ones(len(frame), dtype=torch.long)
    image = torch.from_numpy(features[[image_index[p] for p in frame["image"]]])
    labels = torch.tensor([label_to_id[label] for label in frame["label"]], dtype=torch.long)
    return TensorDataset(tokens, lengths, image, labels)
