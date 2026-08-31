"""
GISR-Net -- GoogLeNet-based Image-to-Scalar Regressor.

PyTorch implementation of the architecture in Table 1 of the
manuscript:

    Input 224x224x3
    Conv1 7x7 /2 + ReLU        -> 112x112x64
    MaxPool-1 3x3 /2           ->  56x56x64
    Conv2 3x3 /1 + ReLU        ->  56x56x192
    MaxPool-2 3x3 /2           ->  28x28x192
    Inception-3a / 3b          ->  28x28x256 / 28x28x480
    MaxPool-3 3x3 /2           ->  14x14x480
    Inception-4a .. 4e         ->  14x14x512 ... 14x14x832
    MaxPool-4 3x3 /2           ->    7x7x832
    Inception-5a / 5b          ->    7x7x832 / 7x7x1024
    Average pooling 7x7        ->    1x1x1024
    Dropout (p = 0.40)
    Fully connected            ->    1x1x1
    Regression output          ->    GICQI scalar

"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torchvision

from config import WEIGHTS_DIR, PRETRAINED_FILES


# --------------------------------------------------------------------------- #
# Local (offline) ImageNet weight loading
# --------------------------------------------------------------------------- #
def load_local_state_dict(key: str):
    """Return an ImageNet state_dict from ROOT/pretrained_weights, or None."""
    fname = PRETRAINED_FILES.get(key)
    if fname is None:
        return None
    path = WEIGHTS_DIR / fname
    if not path.exists():
        # tolerate any file whose name starts with the backbone key
        cands = sorted(WEIGHTS_DIR.glob(f"{key}*.pth")) + sorted(WEIGHTS_DIR.glob(f"{key}*.pt"))
        if not cands:
            return None
        path = cands[0]
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def _torchvision_pretrained(key: str):
    """Ask torchvision for ImageNet weights, tolerating old and new APIs.

    Returns the constructed model, or None if the weights could not be obtained
    (for example when the machine has no access to the PyTorch model zoo).
    """
    fn = {"googlenet": torchvision.models.googlenet,
          }[key]
    extra = {"aux_logits": False} if key == "googlenet" else {}
    for kwargs in ({"weights": "IMAGENET1K_V1"}, {"pretrained": True}):
        try:
            return fn(**kwargs, **extra)
        except Exception:
            continue
    return None


def build_backbone(key: str, pretrained: bool):
    """Instantiate a torchvision backbone.

    Transfer-learning weights are obtained in this order:
      1. torchvision's own ImageNet weights (downloaded and cached automatically);
      2. a state_dict stored in ROOT/pretrained_weights (offline machines);
      3. random initialisation, with a loud warning.
    """
    ctor = {
        "googlenet": lambda: torchvision.models.googlenet(
            weights=None, aux_logits=False, init_weights=True),
    }[key]

    if not pretrained:
        return ctor()

    model = _torchvision_pretrained(key)
    if model is not None:
        return model

    model = ctor()
    sd = load_local_state_dict(key)
    if sd is None:
        warnings.warn(
            f"[{key}] ImageNet weights could not be downloaded and none were found in "
            f"{WEIGHTS_DIR}. Falling back to random initialisation -- this is the RI "
            "condition, not transfer learning.", RuntimeWarning)
    else:
        missing, _ = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "aux" not in m]
        if missing:
            warnings.warn(f"[{key}] missing keys on load: {missing[:6]} ...")
    return model


# --------------------------------------------------------------------------- #
# Regression heads
# --------------------------------------------------------------------------- #
def make_head(in_features: int, variant: str = "linear", p_drop: float = 0.40) -> nn.Module:
    if variant == "linear":
        return nn.Sequential(nn.Dropout(p_drop), nn.Linear(in_features, 1))
    if variant == "mlp":
        return nn.Sequential(
            nn.Dropout(p_drop), nn.Linear(in_features, 256), nn.ReLU(inplace=True),
            nn.Dropout(p_drop / 2), nn.Linear(256, 1))
    raise ValueError(f"unknown head variant '{variant}'")


# --------------------------------------------------------------------------- #
# GISR-Net
# --------------------------------------------------------------------------- #
class GISRNet(nn.Module):
    """GoogLeNet backbone with the classifier/softmax replaced by a regression layer."""

    name = "GISR-Net"

    def __init__(self, pretrained: bool = True, head: str = "linear", p_drop: float = 0.40):
        super().__init__()
        self.backbone = build_backbone("googlenet", pretrained)
        in_f = self.backbone.fc.in_features            # 1024
        self.backbone.dropout = nn.Identity()          # dropout lives in the head
        self.backbone.fc = nn.Identity()
        self.head = make_head(in_f, head, p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)                           # (B, 1024) after 7x7 average pooling
        return self.head(f).squeeze(-1)


def count_parameters(model: nn.Module) -> float:
    """Trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def build_model(arch: str = "gisrnet", pretrained: bool = True, head: str = "linear"):
    """Construct GISR-Net.  `arch` is kept for checkpoint compatibility."""
    arch = arch.lower()
    if arch in ("gisrnet", "gisr-net", "googlenet"):
        return GISRNet(pretrained=pretrained, head=head)
    raise ValueError(f"unknown architecture '{arch}'")


if __name__ == "__main__":
    m = GISRNet(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    print(m.name, "output", m(x).shape, f"{count_parameters(m):.2f} M params")
    for h in ("linear", "mlp", "sigmoid"):
        mm = GISRNet(pretrained=False, head=h)
        print(f"  head={h:8s} -> {mm(x).shape}, {count_parameters(mm):.2f} M")
