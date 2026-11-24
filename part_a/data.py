"""Loading the prepared informativeness splits, tokenisation and the image-feature cache."""
import html
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch

from . import config as C

PAD, UNK = 0, 1
_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_TOKEN = re.compile(r"<url>|<user>|[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def load_split(split):
    """split in {train, valid, test}. IDs stay strings, as the brief requires."""
    # Default CSV quoting matches the csv.DictWriter used by prepare_crisismmd.py.
    df = pd.read_csv(C.PREPARED / f"{split}.tsv", sep="\t", dtype=str, keep_default_na=False)
    df["y"] = df["label"].map({lab: i for i, lab in enumerate(C.LABELS)})
    assert df["y"].notna().all(), f"Unexpected labels in {split}"
    return df


def _fix_mojibake(text):
    # Some tweets were stored as UTF-8 bytes decoded as cp1252 (e.g. "Weâ€™re").
    if "â" in text or "Ã" in text:
        try:
            return text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return text


def tokenize(text):
    text = html.unescape(_fix_mojibake(text)).lower().replace("’", "'").replace("‘", "'")
    text = _URL.sub(" <url> ", text)
    text = _USER.sub(" <user> ", text)
    text = text.replace("#", " ")        # keep hashtag words, drop the symbol
    text = re.sub(r"^rt\b", " ", text)  # retweet marker carries no content
    return _TOKEN.findall(text)


class Vocab:
    """Built from training tweets only (no validation/test leakage)."""

    def __init__(self, texts, min_freq):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        self.itos = ["<pad>", "<unk>"] + sorted(w for w, c in counts.items() if c >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, texts, max_tokens):
        ids = torch.full((len(texts), max_tokens), PAD, dtype=torch.long)
        lengths = torch.ones(len(texts), dtype=torch.long)
        for i, t in enumerate(texts):
            toks = [self.stoi.get(w, UNK) for w in tokenize(t)][:max_tokens] or [UNK]
            ids[i, :len(toks)] = torch.tensor(toks)
            lengths[i] = len(toks)
        return ids, lengths


def load_image_features():
    if not C.FEATURE_CACHE.exists():
        raise FileNotFoundError(
            f"Missing {C.FEATURE_CACHE}. Run: python -m part_a.extract_features")
    z = np.load(C.FEATURE_CACHE, allow_pickle=False)
    return {p: i for i, p in enumerate(z["paths"].tolist())}, torch.from_numpy(z["features"])


def build_tensors(fraction=1.0, seed=0):
    """Returns vocab and {split: dict(ids, lengths, img, y, df)}.

    fraction < 1 subsamples training data only (debug runs); validation/test stay whole.
    """
    index, feats = load_image_features()
    splits = {s: load_split(s) for s in ("train", "valid", "test")}
    if fraction < 1.0:
        splits["train"] = splits["train"].sample(frac=fraction, random_state=seed)
        splits["train"] = splits["train"].sort_index().reset_index(drop=True)
    vocab = Vocab(splits["train"]["tweet_text"], C.HPARAMS["min_freq"])
    out = {}
    for name, df in splits.items():
        ids, lengths = vocab.encode(df["tweet_text"].tolist(), C.HPARAMS["max_tokens"])
        missing = [p for p in df["image"] if p not in index]
        if missing:
            raise KeyError(f"{len(missing)} {name} images missing from feature cache")
        img = feats[[index[p] for p in df["image"]]].float()
        out[name] = {"ids": ids, "lengths": lengths, "img": img,
                     "y": torch.tensor(df["y"].values, dtype=torch.long), "df": df}
    return vocab, out
