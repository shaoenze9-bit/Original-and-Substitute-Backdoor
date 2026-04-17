import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ==========================================
# 0. 基础设置
# ==========================================
def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(2026)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 新增：统一保存目录 =====
SAVE_DIR = "/content/clean_model_results"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "clean_resnet18_cifar10_best.pth")
LAST_MODEL_PATH = os.path.join(SAVE_DIR, "clean_resnet18_cifar10_last.pth")
SUMMARY_PATH = os.path.join(SAVE_DIR, "clean_training_summary.txt")

BATCH_SIZE = 128
EPOCHS = 60
LR = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_WORKERS = 2


# ==========================================
# 1. CIFAR-10 用 ResNet-18
# ==========================================
def build_resnet18_for_cifar10(num_classes=10):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ==========================================
# 2. 数据
# ==========================================
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_tf
)

test_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_tf
)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ==========================================
# 3. 训练 / 测试函数
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


# ==========================================
# 4. 主训练
# ==========================================
def main():
    model = build_resnet18_for_cifar10().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=False
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 75],
        gamma=0.1
    )

    best_ba = 0.0
    best_epoch = -1

    history = []

    print(f">>> Device: {DEVICE}")
    print(f">>> Save dir: {SAVE_DIR}")
    print(f">>> Best model path: {BEST_MODEL_PATH}")
    print(f">>> Last model path: {LAST_MODEL_PATH}")
    print("-" * 110)
    print(f"{'Epoch':<8}{'LR':<12}{'TrainLoss':<14}{'TrainAcc':<14}{'TestLoss':<14}{'BA(TestAcc)':<14}")
    print("-" * 110)

    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        if test_acc > best_ba:
            best_ba = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        print(
            f"{epoch+1:<8}"
            f"{current_lr:<12.6f}"
            f"{train_loss:<14.4f}"
            f"{train_acc:<14.4f}"
            f"{test_loss:<14.4f}"
            f"{test_acc:<14.4f}"
        )

    # ===== 额外保存最后一个 epoch 的模型 =====
    torch.save(model.state_dict(), LAST_MODEL_PATH)

    # ===== 保存训练摘要 =====
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("Clean ResNet-18 Training Summary (CIFAR-10)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Save dir: {SAVE_DIR}\n")
        f.write(f"Best model path: {BEST_MODEL_PATH}\n")
        f.write(f"Last model path: {LAST_MODEL_PATH}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Initial LR: {LR}\n")
        f.write(f"Momentum: {MOMENTUM}\n")
        f.write(f"Weight decay: {WEIGHT_DECAY}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best BA(Test Acc): {best_ba:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Epoch':<8}{'LR':<12}{'TrainLoss':<14}{'TrainAcc':<14}{'TestLoss':<14}{'TestAcc':<14}\n")
        f.write("-" * 80 + "\n")
        for row in history:
            f.write(
                f"{row['epoch']:<8}"
                f"{row['lr']:<12.6f}"
                f"{row['train_loss']:<14.4f}"
                f"{row['train_acc']:<14.4f}"
                f"{row['test_loss']:<14.4f}"
                f"{row['test_acc']:<14.4f}\n"
            )

    print("-" * 110)
    print(f">>> Best BA(Test Acc): {best_ba:.4f} at epoch {best_epoch}")
    print(f">>> Best clean model saved to: {BEST_MODEL_PATH}")
    print(f">>> Last epoch model saved to: {LAST_MODEL_PATH}")
    print(f">>> Training summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
