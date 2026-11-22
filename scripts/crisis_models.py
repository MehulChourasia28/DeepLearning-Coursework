"""Text preprocessing, models, training loop and metrics for CrisisMMD Parts A and B.

Images arrive as cached 512-d frozen ResNet18 features; text as token ids from a
training-only vocabulary. Every model ends in the same MLP head so that text-only,
image-only and fusion variants differ only in their input branches.
"""
import csv
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

PAD_ID, UNK_ID = 0, 1
_URL = re.compile(r"https?://\S+")
_USER = re.compile(r"@\w+")
_TOKEN = re.compile(r"<url>|<user>|[a-z0-9]+(?:'[a-z]+)?|[!?]")


# --------------------------------------------------------------------------- data
def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def tokenize(text):
    """Lower-case; URLs -> <url>, @mentions -> <user>; drop '#' and a leading RT."""
    text = _USER.sub(" <user> ", _URL.sub(" <url> ", text.lower()))
    tokens = _TOKEN.findall(text)
    return tokens[1:] if tokens[:1] == ["rt"] else tokens


class Vocabulary:
    """Training-only vocabulary; ids 0 and 1 are <pad> and <unk>."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=2):
        counts = Counter(token for tokens in token_lists for token in tokens)
        kept = sorted((t for t, c in counts.items() if c >= min_freq), key=lambda t: (-counts[t], t))
        return cls(["<pad>", "<unk>", *kept])

    def __len__(self):
        return len(self.itos)

    def encode(self, token_lists, max_tokens):
        """Return padded ids [N, max_tokens], true lengths [N] and truncation count."""
        ids = torch.zeros(len(token_lists), max_tokens, dtype=torch.long)
        lengths = torch.ones(len(token_lists), dtype=torch.long)
        truncated = 0
        for i, tokens in enumerate(token_lists):
            truncated += len(tokens) > max_tokens
            row = [self.stoi.get(t, UNK_ID) for t in tokens[:max_tokens]] or [UNK_ID]
            ids[i, :len(row)] = torch.tensor(row)
            lengths[i] = len(row)
        return ids, lengths, truncated


def load_task_split(tsv_path, cache_path, label_to_id):
    """TSV rows plus their cached image features, with an alignment check."""
    rows = read_tsv(tsv_path)
    cache = torch.load(cache_path, weights_only=True)
    index = {image_id: i for i, image_id in enumerate(cache["image_ids"])}
    missing = [r["image_id"] for r in rows if r["image_id"] not in index]
    if missing:
        raise ValueError(f"{len(missing)} images missing from {cache_path}, e.g. {missing[:3]}")
    order = torch.tensor([index[r["image_id"]] for r in rows])
    return {
        "rows": rows,
        "image": cache["features"][order].float(),
        "labels": torch.tensor([label_to_id[r["label"]] for r in rows]),
    }


# ------------------------------------------------------------------------- models
class TextCNN(nn.Module):
    """Kim-style CNN: parallel 1-D convolutions, masked max-pool over time."""

    def __init__(self, vocab_size, embed_dim=128, kernel_sizes=(2, 3, 4, 5), filters=64, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, filters, k) for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = filters * len(kernel_sizes)

    def forward(self, ids, lengths):
        x = self.embedding(ids).transpose(1, 2)                      # [B, E, T]
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))                                      # [B, F, T-k+1]
            valid = (lengths - conv.kernel_size[0] + 1).clamp(min=1)  # windows inside the text
            mask = torch.arange(h.size(2), device=h.device)[None, :] < valid[:, None]
            pooled.append(h.masked_fill(~mask[:, None, :], float("-inf")).max(dim=2).values)
        return self.dropout(torch.cat(pooled, dim=1))


class BiLSTMEncoder(nn.Module):
    """One-layer bidirectional LSTM; final forward and backward states concatenated."""

    def __init__(self, vocab_size, embed_dim=128, hidden_size=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = 2 * hidden_size

    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)                           # [2, B, H]
        return self.dropout(torch.cat([hidden[0], hidden[1]], dim=1))


class ImageBranch(nn.Module):
    """Projects cached frozen ResNet18 features; the ResNet itself is never trained."""

    def __init__(self, in_dim=512, out_dim=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))
        self.output_dim = out_dim

    def forward(self, features):
        return self.net(features)


class CrisisClassifier(nn.Module):
    """[text branch] and/or [image branch] -> concatenation -> shared MLP head."""

    def __init__(self, num_classes, text_encoder=None, image_branch=None, hidden=128, dropout=0.5):
        super().__init__()
        if text_encoder is None and image_branch is None:
            raise ValueError("Need at least one branch")
        self.text_encoder, self.image_branch = text_encoder, image_branch
        in_dim = sum(b.output_dim for b in (text_encoder, image_branch) if b is not None)
        self.head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, num_classes))

    def forward(self, batch):
        parts = []
        if self.text_encoder is not None:
            parts.append(self.text_encoder(batch["ids"], batch["lengths"]))
        if self.image_branch is not None:
            parts.append(self.image_branch(batch["image"]))
        return self.head(torch.cat(parts, dim=1))


HYPERPARAMETERS = {
    "embed_dim": 128, "textcnn_kernel_sizes": [2, 3, 4, 5], "textcnn_filters": 64,
    "bilstm_hidden_size": 128, "image_projection_dim": 256, "head_hidden": 128,
    "dropout": 0.5,
}


def build_model(kind, vocab_size, num_classes, hp=HYPERPARAMETERS):
    text = {
        "textcnn": lambda: TextCNN(vocab_size, hp["embed_dim"], tuple(hp["textcnn_kernel_sizes"]),
                                   hp["textcnn_filters"], hp["dropout"]),
        "bilstm": lambda: BiLSTMEncoder(vocab_size, hp["embed_dim"], hp["bilstm_hidden_size"], hp["dropout"]),
    }
    image = lambda: ImageBranch(512, hp["image_projection_dim"], hp["dropout"])
    builders = {
        "text_textcnn": lambda: dict(text_encoder=text["textcnn"]()),
        "text_bilstm": lambda: dict(text_encoder=text["bilstm"]()),
        "image_mlp": lambda: dict(image_branch=image()),
        "fusion_textcnn": lambda: dict(text_encoder=text["textcnn"](), image_branch=image()),
        "fusion_bilstm": lambda: dict(text_encoder=text["bilstm"](), image_branch=image()),
    }
    return CrisisClassifier(num_classes, hidden=hp["head_hidden"], dropout=hp["dropout"], **builders[kind]())


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------- training
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _batches(n, batch_size, generator=None):
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def _take(data, index):
    return {k: data[k][index] for k in ("ids", "lengths", "image", "labels")}


def train_one_epoch(model, data, optimizer, *, batch_size, generator, run, epoch):
    model.train()
    total, seen = 0.0, 0
    for batch_index, index in enumerate(_batches(len(data["labels"]), batch_size, generator), start=1):
        batch = _take(data, index.to(data["labels"].device))
        loss = F.cross_entropy(model(batch), batch["labels"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = loss.item()
        total += value * len(index)
        seen += len(index)
        run.log_batch(value, optimizer, epoch=epoch, batch=batch_index, examples=len(index))
    return total / seen


@torch.no_grad()
def predict(model, data, batch_size=256):
    """Mean cross-entropy and class probabilities, in data order."""
    model.eval()
    probs, total = [], 0.0
    for index in _batches(len(data["labels"]), batch_size):
        batch = _take(data, index.to(data["labels"].device))
        logits = model(batch)
        total += F.cross_entropy(logits, batch["labels"], reduction="sum").item()
        probs.append(logits.softmax(dim=1).cpu())
    return total / len(data["labels"]), torch.cat(probs)


def fit(model, train_data, valid_data, *, run, checkpoint_path, num_classes, lr, batch_size,
        max_epochs, patience, seed):
    """Adam training with validation macro-F1 model selection and early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    bad_epochs = 0
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_data, optimizer, batch_size=batch_size,
                                     generator=generator, run=run, epoch=epoch)
        valid_loss, valid_probs = predict(model, valid_data)
        valid_f1 = classification_metrics(valid_data["labels"].cpu(), valid_probs.argmax(1),
                                          num_classes)["macro_f1"]
        run.log_epoch(epoch, train_loss=train_loss, valid_loss=valid_loss,
                      valid_macro_f1=valid_f1, optimizer=optimizer)
        if run.summary["selected_epoch"] == epoch:
            bad_epochs = 0
            torch.save({"epoch": epoch, "model": model.state_dict()}, checkpoint_path)
            run.update(checkpoint_path=str(checkpoint_path))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                run.early_stop()
                break


# ----------------------------------------------------------------------- metrics
def confusion_matrix(labels, preds, num_classes):
    labels, preds = torch.as_tensor(labels), torch.as_tensor(preds)
    return torch.bincount(labels * num_classes + preds, minlength=num_classes ** 2).reshape(
        num_classes, num_classes)


def classification_metrics(labels, preds, num_classes):
    """Rows of the confusion matrix are true classes. Undefined ratios count as 0."""
    cm = confusion_matrix(labels, preds, num_classes).double()
    tp = cm.diag()
    support, predicted = cm.sum(1), cm.sum(0)
    precision = torch.where(predicted > 0, tp / predicted.clamp(min=1), torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support.clamp(min=1), torch.zeros_like(tp))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall).clamp(min=1e-12),
                     torch.zeros_like(tp))
    return {
        "accuracy": (tp.sum() / cm.sum()).item(),
        "macro_f1": f1.mean().item(),
        "weighted_f1": ((f1 * support).sum() / support.sum()).item(),
        "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
        "support": support.long().tolist(),
        "confusion_matrix": cm.long().tolist(),
    }


def save_predictions(path, rows, labels, probs, id_to_label, split):
    """Per-example predictions with all class probabilities."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = probs.argmax(1)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "tweet_id", "image_id", "label", "prediction",
                         *[f"prob_{id_to_label[i]}" for i in range(probs.size(1))]])
        for row, y, p, pr in zip(rows, labels.tolist(), preds.tolist(), probs.tolist()):
            writer.writerow([split, row["tweet_id"], row["image_id"], id_to_label[y], id_to_label[p],
                             *[f"{v:.6f}" for v in pr]])
    return str(path)
