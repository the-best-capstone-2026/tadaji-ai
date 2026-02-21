import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.model import load_model

# ===== device 설정 =====
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ===== 데이터 경로 =====
DATA_DIR = "./data"  # data/training, data/validation 구조

# ===== 전처리 =====
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

transform_val = transform_train

# ===== Dataset =====
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "training"),
    transform=transform_train
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "validation"),
    transform=transform_val
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ===== 모델 =====
model = load_model(num_classes=11, device=device)
print(train_dataset.classes)

# backbone freeze
for param in model.parameters():
    param.requires_grad = False

# fc만 학습
for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# ===== 학습 =====
EPOCHS = 10
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader)}")

    # ===== 검증 =====
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")

    # best model 저장
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pt")
        print("Best model saved!")

print("Training Finished")