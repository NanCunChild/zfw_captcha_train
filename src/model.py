# model.py
"""CRNN models for captcha recognition.

Four variants are exposed through :func:`build_model`:

* ``tiny``   ~ 1 MB  (light custom CNN + small Bi-LSTM)
* ``small``  ~ 3 MB  (slightly wider custom CNN + Bi-LSTM)
* ``medium`` ~ 10 MB (deeper custom CNN + 2-layer Bi-LSTM)
* ``large``  unlimited (ResNet-18 backbone + 2-layer Bi-LSTM, original model)

All variants output ``(sequence_length, batch, num_classes)`` log-its where
``num_classes`` already includes the CTC blank token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# ---------------------------------------------------------------------------
# Lightweight CRNN (used for tiny / small / medium variants)
# ---------------------------------------------------------------------------

# Each entry roughly targets a saved-state-dict size in float32:
#   tiny   ~  1 MB
#   small  ~  3 MB
#   medium ~ 10 MB
LIGHT_MODEL_CONFIGS = {
    "tiny": {
        "channels": [16, 32, 64, 128],
        "lstm_hidden": 96,
        "lstm_layers": 1,
        "lstm_dropout": 0.0,
    },
    "small": {
        "channels": [32, 64, 128, 256],
        "lstm_hidden": 128,
        "lstm_layers": 1,
        "lstm_dropout": 0.0,
    },
    "medium": {
        "channels": [32, 64, 128, 256, 256],
        "lstm_hidden": 192,
        "lstm_layers": 2,
        "lstm_dropout": 0.2,
    },
}


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class LightCRNN(nn.Module):
    """A compact CRNN whose capacity is chosen by ``variant``."""

    def __init__(self, num_classes: int, variant: str = "tiny") -> None:
        super().__init__()
        if variant not in LIGHT_MODEL_CONFIGS:
            raise ValueError(
                f"Unknown light variant: {variant!r}. "
                f"Expected one of {list(LIGHT_MODEL_CONFIGS)}."
            )

        cfg = LIGHT_MODEL_CONFIGS[variant]
        channels = cfg["channels"]
        hidden = cfg["lstm_hidden"]
        num_layers = cfg["lstm_layers"]
        dropout = cfg["lstm_dropout"]

        layers: list[nn.Module] = []
        in_ch = 3
        n = len(channels)
        for i, out_ch in enumerate(channels):
            layers.append(_conv_block(in_ch, out_ch))
            if i < n - 1:
                # First two pools shrink both dims; later pools shrink only
                # height so we keep enough width for the CTC sequence.
                if i < 2:
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)
        # Collapse any remaining height into the channel-time view.
        self.collapse = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=channels[-1],
            hidden_size=hidden,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                       # (B, C, H, W)
        x = self.collapse(x)                  # (B, C, 1, W)
        x = x.squeeze(2)                      # (B, C, W)
        x = x.permute(2, 0, 1).contiguous()   # (W, B, C)
        x, _ = self.rnn(x)                    # (W, B, 2*hidden)
        x = self.fc(x)                        # (W, B, num_classes)
        return x


# ---------------------------------------------------------------------------
# Original CRNN (ResNet-18 backbone) -- kept as the "large" / unlimited variant
# ---------------------------------------------------------------------------


class CRNN(nn.Module):
    """ResNet-18 backbone + 2-layer Bi-LSTM. Original full-size model."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            resnet = models.resnet18(weights=weights)
        except Exception:
            # Network unreachable / SSL issue: fall back to random init so the
            # script still works in offline or sandboxed environments.
            resnet = models.resnet18(weights=None)
        # Drop the final AvgPool + FC layers.
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.collapse = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False,
        )
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                                # (B, 512, H, W)
        b, c, h, w = x.shape
        # Flatten H*W into the time dimension so the sequence is long enough
        # for CTC alignment (ResNet-18 reduces a 34x90 input to 2x3, which
        # would otherwise leave only 3 time-steps).
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, H*W, C)
        x = x.permute(1, 0, 2).contiguous()             # (H*W, B, C)
        x, _ = self.rnn(x)                              # (H*W, B, 2*hidden)
        x = self.fc(x)                                  # (H*W, B, num_classes)
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

VARIANTS = ("tiny", "small", "medium", "large")


def build_model(variant: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a CRNN for the requested size ``variant``.

    Parameters
    ----------
    variant : str
        One of ``tiny``, ``small``, ``medium``, ``large``.
    num_classes : int
        Number of output classes (must already include the CTC blank).
    pretrained : bool
        Only meaningful for ``large``; if ``True`` (default) try to download
        ImageNet-pretrained ResNet-18 weights, falling back to random init when
        the network is unreachable.
    """
    variant = variant.lower()
    if variant == "large":
        return CRNN(num_classes=num_classes, pretrained=pretrained)
    if variant in LIGHT_MODEL_CONFIGS:
        return LightCRNN(num_classes=num_classes, variant=variant)
    raise ValueError(
        f"Unknown model variant: {variant!r}. Expected one of {VARIANTS}."
    )


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
