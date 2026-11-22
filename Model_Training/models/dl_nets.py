"""PyTorch network definitions: 6 deep architectures + 6 CNN hybrids.

All networks take a normalised window batch (B, C, T) and return class logits (B, K).
Every architecture is defined exactly once; hybrids differ from the pure models by sharing the same
convolutional front-end (``ConvFrontEnd``) that turns the signal into a shorter sequence of learned features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, k, stride=1):
    return nn.Sequential(nn.Conv1d(cin, cout, k, stride, padding=k // 2, bias=False),
                         nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


class ConvFrontEnd(nn.Module):
    """(B, C, T) -> (B, width, T // 4): two conv+pool stages shared by all hybrid models."""

    def __init__(self, c_in, width=32):
        super().__init__()
        self.net = nn.Sequential(conv_bn(c_in, width, 5), nn.MaxPool1d(2), conv_bn(width, width, 5), nn.MaxPool1d(2))
        self.width = width

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- pure models
class CNN1D(nn.Module):
    """Channels as input maps, convolution over time; global avg+max pooling; 64-d embedding."""

    def __init__(self, c_in, n_samples, n_classes, width=32, dropout=0.3, emb=64):
        super().__init__()
        self.features = nn.Sequential(conv_bn(c_in, width, 7), nn.MaxPool1d(2),
                                      conv_bn(width, 2 * width, 5), nn.MaxPool1d(2),
                                      conv_bn(2 * width, 2 * width, 3))
        self.fc = nn.Sequential(nn.Linear(4 * width, emb), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.out = nn.Linear(emb, n_classes)
        self.emb_dim = emb

    def embed(self, x):
        h = self.features(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.fc(h)

    def forward(self, x):
        return self.out(self.embed(x))


class CNN2D(nn.Module):
    """The window is a 1 x C x T image: kernels span neighbouring electrodes *and* time (spatial-temporal)."""

    def __init__(self, c_in, n_samples, n_classes, width=16, dropout=0.3):
        super().__init__()

        def block(ci, co, k, pool):
            return nn.Sequential(nn.Conv2d(ci, co, k, padding=(k[0] // 2, k[1] // 2), bias=False),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True), nn.MaxPool2d(pool))

        self.features = nn.Sequential(block(1, width, (3, 7), (1, 2)), block(width, 2 * width, (3, 5), (2, 2)),
                                      block(2 * width, 2 * width, (3, 3), (1, 1)), nn.AdaptiveAvgPool2d((1, 4)))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2 * width * 4, 64),
                                  nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)))


class RecurrentNet(nn.Module):
    """LSTM / BiLSTM / GRU on the (lightly down-sampled) raw sequence; last hidden state -> classifier."""

    def __init__(self, c_in, n_samples, n_classes, cell="lstm", bidirectional=False, hidden=64,
                 layers=1, dropout=0.3, input_downsample=2):
        super().__init__()
        self.pool = nn.AvgPool1d(input_downsample) if input_downsample > 1 else nn.Identity()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(c_in, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional,
                       dropout=dropout if layers > 1 else 0.0)
        self.dirs = 2 if bidirectional else 1
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * self.dirs, n_classes))

    def forward(self, x):
        h = self.pool(x).transpose(1, 2)                    # (B, T', C)
        _, state = self.rnn(h)
        h_n = state[0] if isinstance(state, tuple) else state      # LSTM returns (h, c)
        last = h_n[-self.dirs:]                              # final layer, each direction
        return self.head(torch.cat(list(last), dim=1))


class TransformerNet(nn.Module):
    """Patch embedding (strided conv) + learned positions + Transformer encoder + mean pooling."""

    def __init__(self, c_in, n_samples, n_classes, d_model=64, heads=4, layers=2, dropout=0.2, n_tokens=20):
        super().__init__()
        patch = max(1, n_samples // n_tokens)
        self.embed = nn.Conv1d(c_in, d_model, patch, stride=patch)
        length = n_samples // patch
        self.pos = nn.Parameter(torch.zeros(1, length, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x):
        h = self.embed(x).transpose(1, 2)
        h = self.encoder(h + self.pos[:, : h.shape[1]])
        return self.head(self.norm(h).mean(1))


# --------------------------------------------------------------------------- hybrids
class CNNRecurrent(nn.Module):
    """CNN front-end -> LSTM / BiLSTM / GRU."""

    def __init__(self, c_in, n_samples, n_classes, cell="lstm", bidirectional=False, width=32, hidden=64, dropout=0.3):
        super().__init__()
        self.front = ConvFrontEnd(c_in, width)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(width, hidden, batch_first=True, bidirectional=bidirectional)
        self.dirs = 2 if bidirectional else 1
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * self.dirs, n_classes))

    def forward(self, x):
        _, state = self.rnn(self.front(x).transpose(1, 2))
        h_n = state[0] if isinstance(state, tuple) else state
        return self.head(torch.cat(list(h_n[-self.dirs:]), dim=1))


class _TCNBlock(nn.Module):
    def __init__(self, ch, k, dilation, dropout):
        super().__init__()
        self.pad = (k - 1) * dilation                       # causal: only past samples
        self.c1 = nn.Conv1d(ch, ch, k, dilation=dilation, bias=False)
        self.b1 = nn.BatchNorm1d(ch)
        self.c2 = nn.Conv1d(ch, ch, k, dilation=dilation, bias=False)
        self.b2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.relu(self.b1(self.c1(F.pad(x, (self.pad, 0))))))
        h = self.drop(F.relu(self.b2(self.c2(F.pad(h, (self.pad, 0))))))
        return F.relu(x + h)


class CNNTCN(nn.Module):
    """CNN front-end -> stack of dilated causal residual blocks (Temporal Convolutional Network)."""

    def __init__(self, c_in, n_samples, n_classes, width=32, dilations=(1, 2, 4), dropout=0.2):
        super().__init__()
        self.front = ConvFrontEnd(c_in, width)
        self.tcn = nn.Sequential(*[_TCNBlock(width, 3, d, dropout) for d in dilations])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * width, n_classes))

    def forward(self, x):
        h = self.tcn(self.front(x))
        return self.head(torch.cat([h[..., -1], h.mean(-1)], dim=1))


class CNNAttention(nn.Module):
    """CNN front-end -> multi-head self-attention (+residual) -> learned attention pooling over time."""

    def __init__(self, c_in, n_samples, n_classes, width=32, heads=4, dropout=0.2):
        super().__init__()
        self.front = ConvFrontEnd(c_in, width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.score = nn.Linear(width, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, n_classes))

    def forward(self, x):
        h = self.front(x).transpose(1, 2)                   # (B, T', W)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm(h + a)
        w = torch.softmax(self.score(h), dim=1)             # (B, T', 1)
        return self.head((w * h).sum(1))


class CNNTransformer(nn.Module):
    """CNN front-end (local features) -> Transformer encoder (global context) -> mean pooling."""

    def __init__(self, c_in, n_samples, n_classes, width=32, d_model=64, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.front = ConvFrontEnd(c_in, width)
        self.proj = nn.Linear(width, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max(1, n_samples // 4), d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x):
        h = self.proj(self.front(x).transpose(1, 2))
        h = self.encoder(h + self.pos[:, : h.shape[1]])
        return self.head(self.norm(h).mean(1))


# --------------------------------------------------------------------------- factory
def build_net(arch: str, c_in: int, n_samples: int, n_classes: int, hp: dict) -> nn.Module:
    dropout = hp.get("dropout", 0.3)
    common = dict(dropout=dropout)
    extra = {k: v for k, v in hp.items() if k not in ("dropout",)}
    table = {
        "cnn1d": lambda: CNN1D(c_in, n_samples, n_classes, **common, **extra),
        "cnn2d": lambda: CNN2D(c_in, n_samples, n_classes, **common, **extra),
        "lstm": lambda: RecurrentNet(c_in, n_samples, n_classes, "lstm", False, **common, **extra),
        "bilstm": lambda: RecurrentNet(c_in, n_samples, n_classes, "lstm", True, **common, **extra),
        "gru": lambda: RecurrentNet(c_in, n_samples, n_classes, "gru", False, **common, **extra),
        "transformer": lambda: TransformerNet(c_in, n_samples, n_classes, **common, **extra),
        "cnn_lstm": lambda: CNNRecurrent(c_in, n_samples, n_classes, "lstm", False, **common, **extra),
        "cnn_gru": lambda: CNNRecurrent(c_in, n_samples, n_classes, "gru", False, **common, **extra),
        "cnn_bilstm": lambda: CNNRecurrent(c_in, n_samples, n_classes, "lstm", True, **common, **extra),
        "cnn_tcn": lambda: CNNTCN(c_in, n_samples, n_classes, **common, **extra),
        "cnn_attention": lambda: CNNAttention(c_in, n_samples, n_classes, **common, **extra),
        "cnn_transformer": lambda: CNNTransformer(c_in, n_samples, n_classes, **common, **extra),
    }
    if arch not in table:
        raise KeyError(f"unknown architecture '{arch}'")
    return table[arch]()
