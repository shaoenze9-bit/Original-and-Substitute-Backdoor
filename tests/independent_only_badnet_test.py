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
# 0. 基础配置
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

AE_MODEL_PATH = "/content/A_WarmStart_TwoStage_StableRotate/ae_A_softgate_psnr38_badnet_ref_ratio45_50_statfixed.pth"
SAVE_DIR = "/content/A_WarmStart_TwoStage_StableRotate"
NEW_RESNET_NAME = "resnet18_B_only_ablation_softgate_psnr38_badnet_ref_ratio45_50_statfixed.pth"

BATCH_SIZE = 128
EPOCHS = 100
POISON_RATE = 0.05
TARGET_LABEL = 0

EPS = 1e-8
TARGET_PSNR = 38.0
TARGET_MSE = 10 ** (-TARGET_PSNR / 10.0)

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 1. A 触发器 AE
# ==========================================
class PoisonAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
# 2. B 触发器
# ==========================================
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

# ==========================================
# 3. A 的频段约束 + 预算构造
# ==========================================
def get_freq_mask_A(H, W, r_low=0.6875, r_high=0.8750, device_target='cpu'):
    fy = torch.fft.fftfreq(H, d=1.0, device=device_target)
    fx = torch.fft.fftfreq(W, d=1.0, device=device_target)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    r_norm = r / (r.max() + 1e-6)
    return (r_norm >= r_low) & (r_norm <= r_high)

def build_A_with_budget(ae, imgs, mask_A):
    gen_A_raw = ae(imgs)
    diff = gen_A_raw - imgs
    fft_diff_filtered = torch.fft.fft2(diff) * mask_A.unsqueeze(0).unsqueeze(1)
    delta = torch.real(torch.fft.ifft2(fft_diff_filtered))
    delta_mse = torch.mean(delta**2, dim=(1, 2, 3), keepdim=True)
    alpha = torch.clamp(torch.sqrt(TARGET_MSE / (delta_mse + EPS)), 0.0, 1.0)
    imgs_A = torch.clamp(imgs + delta * alpha, 0.0, 1.0)
    return imgs_A

# ==========================================
# 4. 数据集
# ==========================================
class AblationPoisonDataset(torch.utils.data.Dataset):
    def __init__(self, root, ae, train=True, mode='train_B_only'):
        self.ds = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=T.ToTensor()
        )
        self.ae = ae.to('cpu').eval()
        self.mode = mode
        self.norm = T.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
        self.mask_A = get_freq_mask_A(32, 32, 0.6875, 0.8750, device_target='cpu')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        img_cpu = img.unsqueeze(0)

        with torch.no_grad():
            # ========= 训练：只允许 B 投毒 =========
            if self.mode == 'train_B_only':
                if random.random() < POISON_RATE:
                    img_cpu = trigger_B(img_cpu)
                    label = TARGET_LABEL

            # ========= 测试四套 =========
            elif self.mode == 'clean':
                pass
            elif self.mode == 'A':
                img_cpu = build_A_with_budget(self.ae, img_cpu, self.mask_A)
            elif self.mode == 'B':
                img_cpu = trigger_B(img_cpu)
            elif self.mode == 'AB':
                img_cpu = build_A_with_budget(self.ae, img_cpu, self.mask_A)
                img_cpu = trigger_B(img_cpu)
            else:
                raise ValueError(f"未知 mode: {self.mode}")

        return self.norm(img_cpu.squeeze(0)), label

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    set_seed(2026)

    # 1) 加载你已经训练好的 A 生成器
    ae = PoisonAE()
    if os.path.exists(AE_MODEL_PATH):
        ae.load_state_dict(torch.load(AE_MODEL_PATH, map_location='cpu'))
        print(f">>> 已加载 CPU 版 AE: {AE_MODEL_PATH}")
    else:
        raise FileNotFoundError(f"未找到 AE 模型: {AE_MODEL_PATH}")

    # 2) 全新分类模型
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(DEVICE)

    # 3) 数据
    train_ds = AblationPoisonDataset('./data', ae, train=True, mode='train_B_only')
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    test_loaders = {
        'BA':     DataLoader(AblationPoisonDataset('./data', ae, train=False, mode='clean'), batch_size=256, num_workers=2),
        'ASR_A':  DataLoader(AblationPoisonDataset('./data', ae, train=False, mode='A'), batch_size=256, num_workers=2),
        'ASR_B':  DataLoader(AblationPoisonDataset('./data', ae, train=False, mode='B'), batch_size=256, num_workers=2),
        'ASR_AB': DataLoader(AblationPoisonDataset('./data', ae, train=False, mode='AB'), batch_size=256, num_workers=2),
    }

    # 4) 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    print(f"\n>>> 启动 B-only 消融训练 | 设备: {DEVICE} | 注入率: {POISON_RATE*100}%")
    print(f">>> A 路径: {AE_MODEL_PATH}")
    print(f">>> A 预算: Target PSNR = {TARGET_PSNR} dB")
    print(f"{'Epoch':<8} | {'BA':<8} | {'ASR_A':<8} | {'ASR_B':<8} | {'ASR_AB':<8}")
    print("-" * 65)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # ===== 评估 =====
        model.eval()
        res = {}
        with torch.no_grad():
            for name, loader in test_loaders.items():
                correct, total = 0, 0
                for x_test, y_test in loader:
                    x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
                    pred = model(x_test).argmax(1)

                    if name == 'BA':
                        correct += (pred == y_test).sum().item()
                        total += y_test.size(0)
                    else:
                        mask = (y_test != TARGET_LABEL)
                        if mask.any():
                            correct += (pred[mask] == TARGET_LABEL).sum().item()
                            total += mask.sum().item()

                res[name] = correct / total if total > 0 else 0.0

        print(f"{epoch+1:03d}      | {res['BA']:.4f} | {res['ASR_A']:.4f} | {res['ASR_B']:.4f} | {res['ASR_AB']:.4f}")

    # 5) 保存
    final_save_path = os.path.join(SAVE_DIR, NEW_RESNET_NAME)
    torch.save(model.state_dict(), final_save_path)
    print(f"\n训练完成！模型存至: {final_save_path}")
