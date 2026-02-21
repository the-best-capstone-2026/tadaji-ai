from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn.functional as F

from src.preprocessing.preprocess import (
    PreprocessConfig,
    preprocess_bytes,
    make_batch,
)
from src.labels import FOOD11_LABELS
from src.model.model import load_model


# =========================
# 1️⃣ Device 설정
# =========================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[Inference] Using device: {device}")


# =========================
# 2️⃣ 모델 로딩
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(BASE_DIR, "..", "best_model.pt")
WEIGHT_PATH = os.path.abspath(WEIGHT_PATH)

model = load_model(num_classes=11, device=device)

if os.path.exists(WEIGHT_PATH):
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    print("[Inference] Loaded best_model.pt successfully")
else:
    print("[Inference] WARNING: best_model.pt not found")

model.eval()


# =========================
# 3️⃣ 예측 함수
# =========================
@torch.no_grad()
def predict_from_bytes(
    image_bytes: bytes,
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
            {"label": "Noodles-Pasta", "prob": 0.04},
            {"label": "Soup", "prob": 0.02}
        ]
    }
    """

    # =========================
    # 전처리 설정
    # =========================
    cfg = PreprocessConfig(
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    # bytes → tensor [3,224,224]
    x = preprocess_bytes(image_bytes, cfg, mode="inference")

    # batch 차원 추가 → [1,3,224,224]
    batch = make_batch(x, device=device)

    # =========================
    # 추론
    # =========================
    logits = model(batch)              # [1, 11]
    probs = F.softmax(logits, dim=1)   # [1, 11]
    probs = probs.squeeze(0)           # [11]

    # =========================
    # Top-K 추출
    # =========================
    topk = min(topk, probs.numel())
    values, indices = torch.topk(probs, k=topk)

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