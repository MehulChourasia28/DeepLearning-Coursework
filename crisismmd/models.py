"""Text encoders, image/fusion classifiers and a majority-class baseline.

Both text encoders output 300-d vectors so the fusion classifier head is identical (same input size and
parameters) whichever text encoder is used; only the text branch changes in the Part A swap.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

IMAGE_DIM = 512
TEXT_DIM = 300


class TextCNN(nn.Module):
    """Kim (2014)-style CNN: parallel 1-D convolutions, ReLU, max-pool over valid positions."""

    def __init__(self, vocab_size, emb_dim=128, num_filters=100, kernel_sizes=(3, 4, 5)):
        super().__init__()
        assert num_filters * len(kernel_sizes) == TEXT_DIM
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes)
        self.out_dim = TEXT_DIM

    def forward(self, tokens, lengths):
        x = self.embedding(tokens).transpose(1, 2)  # (batch, emb, time)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))  # (batch, filters, time - k + 1)
            k = conv.kernel_size[0]
            # A window starting at t is valid if it lies inside the tweet; keep at least one window.
            valid = (lengths - k + 1).clamp(min=1)
            mask = torch.arange(h.size(2), device=h.device)[None, :] < valid[:, None]
            pooled.append(h.masked_fill(~mask[:, None, :], float("-inf")).amax(dim=2))
        return torch.cat(pooled, dim=1)


class BiLSTMEncoder(nn.Module):
    """One-layer bidirectional LSTM; output is the final forward and backward hidden states."""

    def __init__(self, vocab_size, emb_dim=128, hidden=TEXT_DIM // 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden

    def forward(self, tokens, lengths):
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: (2, batch, hidden)
        return torch.cat([h_n[0], h_n[1]], dim=1)


class Classifier(nn.Module):
    """Text-only, image-only or concatenation fusion, all sharing the same MLP head design."""

    def __init__(self, num_classes, text_encoder=None, use_image=False, hidden=256, dropout=0.5):
        super().__init__()
        assert text_encoder is not None or use_image
        self.text_encoder, self.use_image = text_encoder, use_image
        in_dim = (text_encoder.out_dim if text_encoder is not None else 0) + (IMAGE_DIM if use_image else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, tokens, lengths, image):
        parts = []
        if self.use_image:
            parts.append(image)
        if self.text_encoder is not None:
            parts.append(self.text_encoder(tokens, lengths))
        return self.head(torch.cat(parts, dim=1))


def build_model(kind, num_classes, vocab_size=None, text_encoder="textcnn", emb_dim=128, hidden=256,
                dropout=0.5):
    """kind: 'text', 'image' or 'fusion'."""
    encoder = None
    if kind in ("text", "fusion"):
        encoder = {"textcnn": TextCNN, "bilstm": BiLSTMEncoder}[text_encoder](vocab_size, emb_dim)
    return Classifier(num_classes, encoder, use_image=kind in ("image", "fusion"), hidden=hidden,
                      dropout=dropout)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
