# src/tests/test_preprocess.py
from PIL import Image
import torch

from src.preprocessing.preprocess import (
    PreprocessConfig,
    preprocess_pil,
    make_batch,
)


def test_preprocess_output_shape():
    cfg = PreprocessConfig(image_size=224)

    # 더미 이미지 생성 (RGB)
    img = Image.new("RGB", (300, 300), color="red")

    x = preprocess_pil(img, cfg, mode="inference")

    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 224, 224)
    assert x.dtype == torch.float32


def test_make_batch_shape():
    cfg = PreprocessConfig(image_size=224)
    img = Image.new("RGB", (300, 300), color="blue")

    x = preprocess_pil(img, cfg)
    batch = make_batch(x)

    assert batch.shape == (1, 3, 224, 224)
