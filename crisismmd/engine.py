"""Training with early stopping on validation macro-F1, and prediction."""
import copy
import time

import numpy as np
import torch
from sklearn.metrics import f1_score


def _batches(loader, device):
    for tokens, lengths, image, labels in loader:
        yield tokens.to(device), lengths, image.to(device), labels.to(device)


def predict(model, loader, device):
    """Class probabilities (N x C numpy) in loader order."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for tokens, lengths, image, _ in _batches(loader, device):
            outputs.append(torch.softmax(model(tokens, lengths, image), dim=1).cpu())
    return torch.cat(outputs).numpy()


def _mean_loss(model, loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for tokens, lengths, image, labels in _batches(loader, device):
            total += criterion(model(tokens, lengths, image), labels).item() * len(labels)
            count += len(labels)
    return total / count


def fit(model, train_loader, val_loader, criterion, optimizer, device, max_epochs=20, patience=3,
        log=print):
    """Train; keep the weights of the epoch with the best validation macro-F1.

    Stops after `patience` consecutive epochs without a strict improvement.
    Returns (best_state_dict, history list, best_epoch, stop_reason).
    """
    val_labels = val_loader.dataset.tensors[3].numpy()
    best_f1, best_epoch, best_state, history, stale = -1.0, 0, None, [], 0
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for tokens, lengths, image, labels in _batches(train_loader, device):
            optimizer.zero_grad()
            loss = criterion(model(tokens, lengths, image), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)
        probs = predict(model, val_loader, device)
        num_classes = probs.shape[1]
        val_pred = probs.argmax(1)
        val_f1 = float(f1_score(val_labels, val_pred, labels=list(range(num_classes)), average="macro",
                                zero_division=0))
        row = {"epoch": epoch, "train_loss": total / count,
               "val_loss": _mean_loss(model, val_loader, criterion, device),
               "val_macro_f1": val_f1, "val_accuracy": float((val_pred == val_labels).mean()),
               "epoch_seconds": time.perf_counter() - started}
        history.append(row)
        improved = val_f1 > best_f1
        if improved:
            best_f1, best_epoch, best_state, stale = val_f1, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        log(f"  epoch {epoch:2d}  train_loss {row['train_loss']:.4f}  val_loss {row['val_loss']:.4f}  "
            f"val_macro_f1 {val_f1:.4f}{'  *' if improved else ''}")
        if stale >= patience:
            stop_reason = "early_stopped"
            break
    return best_state, history, best_epoch, stop_reason


def class_weights(labels, num_classes):
    """Inverse-frequency weights from training labels: n / (C * count_c)."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    return torch.tensor(len(labels) / (num_classes * np.maximum(counts, 1)), dtype=torch.float32)
