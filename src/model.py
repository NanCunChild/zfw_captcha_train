# model.py
"""Compact pure-CNN models for fixed-length 4-digit captcha recognition.

The captcha task is constrained: exactly 4 digits, no character distortion,
only rotation + noise + line interference. An RNN is overkill, so this module
exposes a family of small CNN-only models that emit ``(B, 4, 10)`` logits --
one classification head per digit position. Training uses the sum of 4
``CrossEntropyLoss`` terms (no CTC).

Variants:

* ``nano``   ~ 21K params  (~85 KB on disk, hard < 100 KB)
* ``tiny``   ~ 96K params  (~385 KB,  < 500 KB)
* ``small``  ~ 196K params (~785 KB)
* ``large``  ~ 1.36M params (~5.4 MB)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


# Each entry defines a stack of 3x3 conv blocks (Conv -> BN -> ReLU). The
# matching ``pools`` list says whether to apply a 2x2 max-pool after that
# block. With a 34x90 input we need exactly 3 pools to land on a ~4x11
# spatial map, which the final adaptive average pool collapses to (1, 4) --
# one feature column per digit.
MODEL_CONFIGS: dict[str, dict[str, Sequence]] = {
    "nano": {
        "channels": [10, 20, 32, 40],
        "pools":    [True, True, True, False],
    },
    "tiny": {
        "channels": [16, 32, 48, 64, 80],
        "pools":    [True, True, True, False, False],
    },
    "small": {
        "channels": [24, 48, 64, 96, 112],
        "pools":    [True, True, True, False, False],
    },
    "medium": {
        "channels": [32, 64, 96, 128, 160, 192],
        "pools":    [True, True, True, False, False, False],
    },
    "large": {
        "channels": [32, 64, 128, 192, 256, 256],
        "pools":    [True, True, True, False, False, False],
    },
}

NUM_POSITIONS = 4   # the captcha always has 4 digits
NUM_DIGITS = 10     # 0-9

VARIANTS = tuple(MODEL_CONFIGS.keys())


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """3x3 Conv -> BN -> ReLU. Bias is dropped because BN absorbs it."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CaptchaCNN(nn.Module):
    """Pure CNN with 4 independent linear heads (one per digit slot)."""

    def __init__(
        self,
        variant: str = "tiny",
        num_positions: int = NUM_POSITIONS,
        num_digits: int = NUM_DIGITS,
    ) -> None:
        super().__init__()
        if variant not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown variant {variant!r}. Expected one of {list(MODEL_CONFIGS)}."
            )

        cfg = MODEL_CONFIGS[variant]
        channels: Sequence[int] = cfg["channels"]
        pools: Sequence[bool] = cfg["pools"]
        if len(channels) != len(pools):
            raise ValueError("`channels` and `pools` must be the same length")

        layers: list[nn.Module] = []
        in_ch = 3
        for out_ch, do_pool in zip(channels, pools):
            layers.append(_conv_block(in_ch, out_ch))
            if do_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        feat_dim = channels[-1]
        # Collapse height to 1 and force exactly ``num_positions`` columns,
        # so each column corresponds to one digit slot in the captcha.
        self.collapse = nn.AdaptiveAvgPool2d((1, num_positions))
        self.heads = nn.ModuleList(
            [nn.Linear(feat_dim, num_digits) for _ in range(num_positions)]
        )
        self.num_positions = num_positions
        self.num_digits = num_digits

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                         # (B, C, H', W')
        x = self.collapse(x)                    # (B, C, 1, num_positions)
        x = x.squeeze(2).permute(0, 2, 1)       # (B, num_positions, C)
        # Apply one Linear head per position. Stacked output: (B, P, num_digits).
        out = torch.stack(
            [head(x[:, i, :]) for i, head in enumerate(self.heads)],
            dim=1,
        )
        return out


def build_model(variant: str, num_classes: int = NUM_DIGITS, **_: object) -> nn.Module:
    """Build a :class:`CaptchaCNN` for the requested ``variant``.

    The legacy ``num_classes`` and ``pretrained`` kwargs are accepted for
    backwards compatibility but ignored: this model fixes the per-position
    output dimension to 10 (digits 0-9) and never loads pretrained weights.
    """
    if num_classes not in (NUM_DIGITS, NUM_DIGITS + 1):
        # We tolerate the old "+1 for CTC blank" call sites, but anything
        # else likely indicates a misconfiguration.
        raise ValueError(
            f"num_classes={num_classes} is not supported. "
            f"This model emits exactly {NUM_DIGITS} classes per digit."
        )
    return CaptchaCNN(variant=variant.lower())


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
