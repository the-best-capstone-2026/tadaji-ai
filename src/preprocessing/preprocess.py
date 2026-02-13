# src/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import io
from PIL import Image

import torch
from torchvision import transforms


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_transform(cfg: PreprocessConfig, mode: str = "inference") -> transforms.Compose:
    """
    mode:
      - "inference": 추론용 (결정된 resize/normalize만)
      - "train": 학습용 (가벼운 augmentation을 넣고 싶으면 여기만 수정)
    """
    if mode not in ("inference", "train"):
        raise ValueError("mode must be 'inference' or 'train'")

    common = [
        transforms.Resize((cfg.image_size, cfg.image_size)),  # 계약: 224x224 고정
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ]

    # 처음엔 train도 inference와 동일하게 두는 걸 추천 (재현성/안정성)
    # augmentation은 성능 실험할 때만 여기에서 추가
    if mode == "train":
        return transforms.Compose(common)

    return transforms.Compose(common)


def _to_rgb_pil(img: Image.Image) -> Image.Image:
    # 모든 입력을 RGB로 통일
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def preprocess_pil(
    img: Image.Image,
    cfg: PreprocessConfig,
    mode: str = "inference",
) -> torch.Tensor:
    """
    return: torch.FloatTensor shape [3, 224, 224]
    """
    img = _to_rgb_pil(img)
    tfm = build_transform(cfg, mode=mode)
    return tfm(img)


def preprocess_bytes(
    image_bytes: bytes,
    cfg: PreprocessConfig,
    mode: str = "inference",
) -> torch.Tensor:
    """
    FastAPI에서 업로드된 파일(bytes) -> Tensor [3,224,224]
    """
    img = Image.open(io.BytesIO(image_bytes))
    return preprocess_pil(img, cfg, mode=mode)


def make_batch(x: torch.Tensor, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """
    x: [3,224,224] -> [1,3,224,224]
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor [C,H,W], got {x.shape}")
    return x.unsqueeze(0).to(device)
