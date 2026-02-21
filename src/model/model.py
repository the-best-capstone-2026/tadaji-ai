# src/model/model.py
import torch
import torch.nn as nn
import torchvision.models as models


def load_model(num_classes: int = 11, device: str = "cpu") -> torch.nn.Module:
    """
    ResNet18 pretrained 모델을 불러오고
    마지막 FC 레이어를 Food-11 클래스 개수에 맞게 수정한다.
    """

    # 1️⃣ ImageNet pretrained ResNet18 불러오기
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2️⃣ 마지막 fully connected layer 수정
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 3️⃣ device 설정
    model = model.to(device)

    return model
