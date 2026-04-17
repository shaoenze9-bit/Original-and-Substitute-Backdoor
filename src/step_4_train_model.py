import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ==========================================
# 1. 基础配置
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

# 路径配置
SAVE_DIR = "/content/A_WarmStart_TwoStage_StableRotate"
AE_MODEL_PATH = "/content/ae_A_softgate_psnr38_badnet_ref_ratio200%_statfixed (1).pth"
NEW_RESNET_NAME = "final_coupled_resnet18_softgate_psnr38_badnet_ref_ratio45_50_statfixed_100.pth"

# 毒化与隐蔽性参数
TARGET_LABEL = 0
POISON_RATE = 0.08
TARGET_PSNR = 38.0
TARGET_MSE = 10 ** (-TARGET_PSNR / 10.0)
EPS = 1e-8

# ==========================================
# 2. 核心组件 (修正：AE 在 CPU 上计算以适配多进程)
# ==========================================
class PoisonAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class BadNetTrigger:
    def __init__(self, patch_size=4, value=1.0, margin=0):
        self.patch_size = patch_size
        self.value = value
        self.margin = margin

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        is_batch = x.dim() == 4
        if not is_batch:
            x = x.unsqueeze(0)

        out = x.clone()
        _, _, H, W = out.shape

        ps = self.patch_size
        m = self.margin

        h_start = H - ps - m
        h_end = H - m
        w_start = W - ps - m
        w_end = W - m

        out[:, :, h_start:h_end, w_start:w_end] = self.value
        out = out.clamp(0.0, 1.0)

        return out if is_batch else out.squeeze(0)


trigger_B = BadNetTrigger(patch_size=4, value=1.0, margin=0)


def get_freq_mask_A(H, W, r_low=0.6875, r_high=0.8750, device_target="cpu"):
    fy = torch.fft.fftfreq(H, d=1.0, device=device_target)
    fx = torch.fft.fftfreq(W, d=1.0, device=device_target)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    r_norm = r / (r.max() + 1e-6)
    return (r_norm >= r_low) & (r_norm <= r_high)


def build_A_with_budget(ae, imgs, mask_A):
    # 此处 imgs 和 mask_A 均在 CPU
    gen_A_raw = ae(imgs)
    diff = gen_A_raw - imgs

    fft_diff = torch.fft.fft2(diff)
    fft_diff_filtered = fft_diff * mask_A.unsqueeze(0).unsqueeze(1)
    delta = torch.real(torch.fft.ifft2(fft_diff_filtered))

    delta_mse = torch.mean(delta**2, dim=(1, 2, 3), keepdim=True)
    alpha = torch.clamp(torch.sqrt(TARGET_MSE / (delta_mse + EPS)), 0.0, 1.0)

    imgs_A = torch.clamp(imgs + delta * alpha, 0.0, 1.0)
    return imgs_A


class CoupledPoisonDataset(torch.utils.data.Dataset):
    def __init__(self, root, ae, train=True, mode="train"):
        self.ds = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=T.ToTensor()
        )
        # 核心修正：AE 放在 CPU，避免多进程 worker 出问题
        self.ae = ae.to("cpu").eval()
        self.mode = mode
        self.norm = T.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
        self.mask_A = get_freq_mask_A(32, 32, 0.6875, 0.8750, device_target="cpu")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        img_cpu = img.unsqueeze(0)  # 保持在 CPU

        with torch.no_grad():
            if self.mode == "train":
                if random.random() < POISON_RATE:
                    strategy = random.choice(["A", "B", "AB"])
                    if strategy == "A":
                        img_cpu = build_A_with_budget(self.ae, img_cpu, self.mask_A)
                    elif strategy == "B":
                        img_cpu = trigger_B(img_cpu)
                    else:
                        img_cpu = build_A_with_budget(self.ae, img_cpu, self.mask_A)
                        img_cpu = trigger_B(img_cpu)
                    label = TARGET_LABEL

            elif self.mode == "A":
                img_cpu = build_A_with_budget(self.ae, img_cpu, self.mask_A)

            elif self.mode == "B":
                img_cpu = trigger_B(img_cpu)

            elif self.mode == "AB":
                img_cpu = build_A_with_budget(self.ae, img_cpu, self.mask_A)
                img_cpu = trigger_B(img_cpu)


        return self.norm(img_cpu.squeeze(0)), label


# ==========================================
# 3. 训练主程序
# ==========================================
if __name__ == "__main__":
    set_seed(2026)

    # 1. 加载本次实验 AE (A)
    ae = PoisonAE()
    if os.path.exists(AE_MODEL_PATH):
        ae.load_state_dict(torch.load(AE_MODEL_PATH, map_location="cpu"))
        print(f">>> 已加载 CPU 版 AE: {AE_MODEL_PATH}")
    else:
        raise FileNotFoundError(f"未找到 AE 模型: {AE_MODEL_PATH}")

    # 2. 初始化全新的 ResNet-18 (GPU)
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(DEVICE)

    # 3. 数据准备
    train_ds = CoupledPoisonDataset("./data", ae, train=True, mode="train")
    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True
    )

    test_loaders = {
        "BA":     DataLoader(CoupledPoisonDataset("./data", ae, train=False, mode="clean"), batch_size=256, num_workers=2),
        "ASR_A":  DataLoader(CoupledPoisonDataset("./data", ae, train=False, mode="A"), batch_size=256, num_workers=2),
        "ASR_B":  DataLoader(CoupledPoisonDataset("./data", ae, train=False, mode="B"), batch_size=256, num_workers=2),
        "ASR_AB": DataLoader(CoupledPoisonDataset("./data", ae, train=False, mode="AB"), batch_size=256, num_workers=2),
    }

    # 4. 优化器 (标准 Scratch 配置)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print(f"\n>>> 启动耦合毒化训练 (softgate_psnr38 + badnet AE) | 设备: {DEVICE} | 注入率: {POISON_RATE*100}%")
    print(f">>> A 路径: {AE_MODEL_PATH}")
    print(f">>> A 预算: Target PSNR = {TARGET_PSNR} dB")
    print(f"{'Epoch':<8} | {'BA':<8} | {'ASR_A':<8} | {'ASR_B':<8} | {'ASR_AB':<8}")
    print("-" * 65)

    for epoch in range(100):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 评估
        model.eval()
        res = {}
        with torch.no_grad():
            for name, loader in test_loaders.items():
                correct, total = 0, 0
                for x_test, y_test in loader:
                    x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
                    pred = model(x_test).argmax(1)

                    if name == "BA":
                        correct += (pred == y_test).sum().item()
                        total += y_test.size(0)
                    else:
                        mask = (y_test != TARGET_LABEL)
                        if mask.any():
                            correct += (pred[mask] == TARGET_LABEL).sum().item()
                            total += mask.sum().item()

                res[name] = correct / total if total > 0 else 0.0

        print(
            f"{epoch+1:03d}      | "
            f"{res['BA']:.4f} | {res['ASR_A']:.4f} | {res['ASR_B']:.4f} | {res['ASR_AB']:.4f}"
        )

    # 5. 保存
    final_save_path = os.path.join(SAVE_DIR, NEW_RESNET_NAME)
    torch.save(model.state_dict(), final_save_path)
    print(f"\n训练完成！模型存至: {final_save_path}")
