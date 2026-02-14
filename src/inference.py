# src/inference.py
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from preprocessing.preprocess import (
    PreprocessConfig,
    preprocess_bytes,
    make_batch,
)
from labels import FOOD11_LABELS


@torch.no_grad()
def predict_from_bytes(
    model: torch.nn.Module,
    image_bytes: bytes,
    device: str = "cpu",
    topk: int = 3,
) -> Dict:
    """
    이미지 bytes를 받아 Food-11 분류 결과를 반환한다.

    return 예시:
    {
        "label": "Rice",
        "confidence": 0.92,
        "topK": [
            {"label": "Rice", "prob": 0.92},
            {"label": "Noodles_Pasta", "prob": 0.04},
            {"label": "Soup", "prob": 0.02}
        ]
    }
    """

    # 전처리 설정 (계약 고정)
    cfg = PreprocessConfig(
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    # bytes → tensor [3,224,224]
    x = preprocess_bytes(image_bytes, cfg, mode="inference")

    # batch 차원 추가 → [1,3,224,224]
    batch = make_batch(x, device=device)

    # 추론
    model.eval()
    logits = model(batch)              # [1, 11]
    probs = F.softmax(logits, dim=1)   # [1, 11]
    probs = probs.squeeze(0)           # [11]

    # Top-K 추출
    topk = min(topk, probs.numel())
    values, indices = torch.topk(probs, k=topk)

    # 결과 포맷팅
    topk_result = []
    for v, i in zip(values, indices):
        topk_result.append(
            {
                "label": FOOD11_LABELS[int(i.item())],
                "prob": float(v.item()),
            }
        )

    result = {
        "label": topk_result[0]["label"],
        "confidence": topk_result[0]["prob"],
        "topK": topk_result,
    }

    return result
