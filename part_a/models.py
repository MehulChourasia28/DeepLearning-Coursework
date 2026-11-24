"""Text encoders, image branch and the shared classifier head.

Both text encoders output 256-d features, so swapping TextCNN <-> BiLSTM in the fusion
model leaves the image branch and the classifier head exactly the same shape.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .data import PAD


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, kernels, filters):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, filters, k) for k in kernels)
        self.out_dim = filters * len(kernels)

    def forward(self, ids, lengths):
        x = self.embed(ids).transpose(1, 2)                      # B x E x T
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))                              # B x F x (T - k + 1)
            # Max-pool only over windows that start on a real token (ignore all-padding
            # windows); a tweet shorter than k keeps its first window.
            k = conv.kernel_size[0]
            valid = torch.clamp(lengths - k + 1, min=1).to(h.device)
            pos = torch.arange(h.size(2), device=h.device)
            mask = pos.unsqueeze(0) >= valid.unsqueeze(1)        # B x (T - k + 1)
            feats.append(h.masked_fill(mask.unsqueeze(1), float("-inf")).amax(2))
        return torch.cat(feats, 1)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.out_dim = 2 * hidden

    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.embed(ids), lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.lstm(packed)                            # h: 2 x B x H
        return torch.cat([h[0], h[1]], 1)                        # final fwd + bwd states


def head(in_dim, hidden, dropout, n_classes=2):
    """Small MLP classifier shared by every learned model."""
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(hidden, n_classes))


class Classifier(nn.Module):
    """kind: text_textcnn | text_bilstm | image_mlp | fusion_textcnn | fusion_bilstm.

    Image branch = the frozen 512-d ResNet18 feature (no trainable parameters here; the
    encoder itself is frozen and cached). Fusion = concatenate [text, image] -> MLP head.
    """

    def __init__(self, kind, vocab_size, hp):
        super().__init__()
        self.kind = kind
        self.use_text = kind != "image_mlp"
        self.use_image = kind.startswith(("image", "fusion"))
        in_dim = 0
        if self.use_text:
            if kind.endswith("textcnn"):
                self.text = TextCNN(vocab_size, hp["embed_dim"], hp["textcnn_kernels"],
                                    hp["textcnn_filters"])
            else:
                self.text = BiLSTM(vocab_size, hp["embed_dim"], hp["lstm_hidden"])
            in_dim += self.text.out_dim
        if self.use_image:
            in_dim += hp["image_dim"]
        self.head = head(in_dim, hp["head_hidden"], hp["dropout"])

    def forward(self, ids, lengths, img):
        parts = []
        if self.use_text:
            parts.append(self.text(ids, lengths))
        if self.use_image:
            parts.append(img)
        return self.head(torch.cat(parts, 1))


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
