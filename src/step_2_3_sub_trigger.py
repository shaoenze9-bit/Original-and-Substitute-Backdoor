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

USE_WARM_START = False
A_WARM_PATH = None

B_MODEL_PATH = "/content/clean_resnet18_cifar10_last.pth"

SAVE_DIR = "/content/A_WarmStart_TwoStage_StableRotate"
SAVE_NAME = "ae_A_softgate_psnr38_badnet_ref_ratio45_50_statfixed.pth"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-4

EPS = 1e-8

RATIO_DENOM_FLOOR = 0.05
RATIO_MAX_CLAMP = 2.0

TARGET_PSNR = 38.0
TARGET_MSE = 10 ** (-TARGET_PSNR / 10.0)

LOWER_RATIO = 4.0
UPPER_RATIO = 5.0
CENTER_RATIO = 4.5

GATE_SIGMA = 0.05
GATE_K = 30.0

COS_CAP_STAGE2 = 0.20
TARGET_ANGLE_STAGE2 = 40.0


# ==========================================
# 2. B 触发器（BadNet）
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
# 3. A 的频段约束
#    改为 0.6875 - 0.8750
# ==========================================
def get_freq_mask_A(H, W, r_low=0.6875, r_high=0.8750):
    fy = torch.fft.fftfreq(H, d=1.0, device=DEVICE)
    fx = torch.fft.fftfreq(W, d=1.0, device=DEVICE)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    r_norm = r / (r.max() + 1e-6)
    return (r_norm >= r_low) & (r_norm <= r_high)


# ==========================================
# 4. 模型
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


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)


def build_resnet18_for_cifar10():
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ==========================================
# 5. 工具函数
# ==========================================
def mse_to_psnr(mse_val):
    return 10.0 * np.log10(1.0 / (mse_val + 1e-10))


def direct_gain_ratio(norm_A, norm_B):
    denom = torch.clamp(norm_B, min=RATIO_DENOM_FLOOR)
    ratio = norm_A / denom
    ratio = torch.clamp(ratio, min=0.0, max=RATIO_MAX_CLAMP)
    return ratio


# ==========================================
# 6. 核心：按 PSNR 预算构造 A
# ==========================================
def build_A_with_budget(ae, imgs, mask_A):
    gen_A_raw = ae(imgs)
    diff = gen_A_raw - imgs

    fft_diff = torch.fft.fft2(diff)
    fft_diff_filtered = fft_diff * mask_A.unsqueeze(0).unsqueeze(1)
    diff_constrained = torch.real(torch.fft.ifft2(fft_diff_filtered))

    delta = diff_constrained
    delta_mse = torch.mean(delta * delta, dim=(1, 2, 3), keepdim=True)

    # 只缩小，不放大：绝不超过 PSNR 预算
    alpha = torch.sqrt(TARGET_MSE / (delta_mse + EPS))
    alpha = torch.clamp(alpha, min=0.0, max=1.0)

    delta_scaled = delta * alpha
    imgs_A = torch.clamp(imgs + delta_scaled, 0.0, 1.0)

    mseA_each = torch.mean((imgs_A - imgs) ** 2, dim=(1, 2, 3))
    return imgs_A, delta_scaled, alpha.squeeze(), mseA_each


# ==========================================
# 7. 软门控函数
# ==========================================
def gaussian_gate(x, center=CENTER_RATIO, sigma=GATE_SIGMA):
    return torch.exp(-((x - center) ** 2) / (2.0 * sigma * sigma + EPS))


def sigmoid_gate(x):
    return torch.sigmoid(x)


# ==========================================
# 8. 主训练
# ==========================================
def run_two_stage_training():
    if not os.path.exists(B_MODEL_PATH):
        raise FileNotFoundError(f"找不到 clean model: {B_MODEL_PATH}")

    base_model = build_resnet18_for_cifar10()
    base_model.load_state_dict(torch.load(B_MODEL_PATH, map_location=DEVICE))
    base_model = base_model.to(DEVICE).eval()

    feat_ext = FeatureExtractor(base_model).to(DEVICE).eval()
    for p in feat_ext.parameters():
        p.requires_grad = False

    ae = PoisonAE().to(DEVICE)
    if USE_WARM_START:
        if A_WARM_PATH is None or (not os.path.exists(A_WARM_PATH)):
            raise FileNotFoundError(f"找不到 warm-start A: {A_WARM_PATH}")
        ae.load_state_dict(torch.load(A_WARM_PATH, map_location=DEVICE))
    ae.train()

    optimizer = optim.Adam(ae.parameters(), lr=LR)

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=T.ToTensor()
    )
    dataloader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    norm_tf = T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )

    mask_A = get_freq_mask_A(32, 32, 0.6875, 0.8750).to(DEVICE)

    print("\n>>> 启动 Soft-Gated Training (clean-reference + single loss)")
    print(f">>> Warm A: {'disabled (random init)' if not USE_WARM_START else A_WARM_PATH}")
    print(f">>> B model: {B_MODEL_PATH}")
    print(f">>> Target PSNR(A): >= {TARGET_PSNR} dB")
    print(f">>> Ratio window: [{LOWER_RATIO:.2f}, {UPPER_RATIO:.2f}] | center={CENTER_RATIO:.3f}")
    print(f">>> Soft gate: sigma={GATE_SIGMA:.3f}, k={GATE_K:.1f}")
    print("-" * 180)

    for epoch in range(EPOCHS):
        ae.train()

        w_mag = 20.0
        w_floor = 20.0
        w_barrier = 1.0
        w_ortho_soft = 20.0
        w_ortho_hard = 60.0 + 40.0 * (epoch / max(1, EPOCHS - 1))
        w_angle_floor = 15.0

        loss_total_list = []
        l_mag_list = []
        l_floor_list = []
        l_barrier_list = []
        l_ortho_soft_list = []
        l_ortho_hard_list = []
        l_angle_floor_list = []

        psnr_list = []
        angle_list = []
        cos_list = []
        ratio_list = []
        normA_list = []
        normB_list = []
        alpha_list = []

        gate_low_list = []
        gate_mid_list = []
        gate_high_list = []

        min_normB_list = []
        p10_normB_list = []
        max_ratio_list = []
        p90_ratio_list = []

        for imgs, _ in dataloader:
            imgs = imgs.to(DEVICE)

            # ---- 构造 A：严格不超过 PSNR 预算 ----
            imgs_A, delta_scaled, alpha, mseA_each = build_A_with_budget(ae, imgs, mask_A)

            # ---- 构造 B ----
            imgs_B = trigger_B(imgs)

            imgs_norm = norm_tf(imgs)
            imgs_A_norm = norm_tf(imgs_A)
            imgs_B_norm = norm_tf(imgs_B)

            with torch.no_grad():
                f_clean = feat_ext(imgs_norm)
                f_B = feat_ext(imgs_B_norm)

            f_A = feat_ext(imgs_A_norm)

            v_A = f_A - f_clean
            v_B = f_B - f_clean

            norm_A = torch.norm(v_A, p=2, dim=1)
            norm_B = torch.norm(v_B, p=2, dim=1)

            cos = torch.sum(v_A * v_B, dim=1) / (norm_A * norm_B + EPS)
            cos = torch.clamp(cos, -1.0, 1.0)

            angle_deg = torch.acos(cos) * (180.0 / np.pi)
            ratio = direct_gain_ratio(norm_A, norm_B)

            # =========================
            # 软门控
            # =========================
            gate_low = sigmoid_gate(GATE_K * (LOWER_RATIO - ratio))
            gate_high = sigmoid_gate(GATE_K * (ratio - UPPER_RATIO))
            gate_mid = gaussian_gate(ratio, CENTER_RATIO, GATE_SIGMA)

            # =========================
            # Loss 分量
            # =========================
            denom_B_detach = torch.clamp(norm_B.detach(), min=RATIO_DENOM_FLOOR)

            l_mag = torch.mean(gate_low * torch.relu(LOWER_RATIO * denom_B_detach - norm_A))
            l_floor = torch.mean(gate_high * torch.relu(norm_A - UPPER_RATIO * denom_B_detach))
            l_barrier = torch.mean((gate_low + gate_high) * ((ratio - CENTER_RATIO) ** 2))

            l_ortho_soft = torch.mean(gate_mid * (cos ** 2))
            l_ortho_hard = torch.mean(gate_mid * torch.relu(cos - COS_CAP_STAGE2) ** 2)
            l_angle_floor = torch.mean(
                gate_mid * (
                    torch.relu(torch.tensor(TARGET_ANGLE_STAGE2, device=DEVICE) - angle_deg) / TARGET_ANGLE_STAGE2
                )
            )

            loss = (
                w_mag * l_mag
                + w_floor * l_floor
                + w_barrier * l_barrier
                + w_ortho_soft * l_ortho_soft
                + w_ortho_hard * l_ortho_hard
                + w_angle_floor * l_angle_floor
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            optimizer.step()

            mean_mseA = mseA_each.mean().item()
            mean_psnrA = mse_to_psnr(mean_mseA)

            loss_total_list.append(loss.item())
            l_mag_list.append(l_mag.item())
            l_floor_list.append(l_floor.item())
            l_barrier_list.append(l_barrier.item())
            l_ortho_soft_list.append(l_ortho_soft.item())
            l_ortho_hard_list.append(l_ortho_hard.item())
            l_angle_floor_list.append(l_angle_floor.item())

            psnr_list.append(mean_psnrA)
            angle_list.append(angle_deg.mean().item())
            cos_list.append(cos.mean().item())
            ratio_list.append(ratio.mean().item() * 100.0)
            normA_list.append(norm_A.mean().item())
            normB_list.append(norm_B.mean().item())
            alpha_list.append(alpha.mean().item())

            gate_low_list.append(gate_low.mean().item())
            gate_mid_list.append(gate_mid.mean().item())
            gate_high_list.append(gate_high.mean().item())

            min_normB_list.append(norm_B.min().item())
            p10_normB_list.append(torch.quantile(norm_B.detach(), 0.10).item())
            max_ratio_list.append(ratio.max().item())
            p90_ratio_list.append(torch.quantile(ratio.detach(), 0.90).item())

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} [SoftGate] | "
            f"w_mag={w_mag:.2f}, w_floor={w_floor:.2f}, w_barrier={w_barrier:.2f}, "
            f"w_ortho_soft={w_ortho_soft:.2f}, w_ortho_hard={w_ortho_hard:.2f}, w_angle_floor={w_angle_floor:.2f} | "
            f"Loss={np.mean(loss_total_list):.4f} | "
            f"L_mag={np.mean(l_mag_list):.6f} | "
            f"L_floor={np.mean(l_floor_list):.6f} | "
            f"L_barrier={np.mean(l_barrier_list):.6f} | "
            f"L_ortho_soft={np.mean(l_ortho_soft_list):.6f} | "
            f"L_ortho_hard={np.mean(l_ortho_hard_list):.6f} | "
            f"L_angle_floor={np.mean(l_angle_floor_list):.6f} | "
            f"PSNR(A)={np.mean(psnr_list):.2f}dB | "
            f"Angle={np.mean(angle_list):.2f}° | "
            f"Cos={np.mean(cos_list):.4f} | "
            f"Ratio={np.mean(ratio_list):.2f}% | "
            f"||A||={np.mean(normA_list):.4f} | "
            f"||B||={np.mean(normB_list):.4f} | "
            f"alpha={np.mean(alpha_list):.4f}"
        )

        print(
            f"  [Gate] low={np.mean(gate_low_list):.4f} | mid={np.mean(gate_mid_list):.4f} | high={np.mean(gate_high_list):.4f}"
        )

        print(
            f"  [Diag] min||B||={np.mean(min_normB_list):.6f} | "
            f"p10||B||={np.mean(p10_normB_list):.6f} | "
            f"p90(ratio)={np.mean(p90_ratio_list):.4f} | "
            f"max(ratio)={np.mean(max_ratio_list):.4f}"
        )

        if np.mean(psnr_list) < TARGET_PSNR - 0.1:
            print("  [警告] PSNR 低于 38 dB，检查预算构造逻辑。")

        if np.mean(ratio_list) < LOWER_RATIO * 100.0:
            print("  [提示] 当前 ratio 仍低于 95%，补模长项正在主导。")
        elif np.mean(ratio_list) > UPPER_RATIO * 100.0:
            print("  [提示] 当前 ratio 高于 100%，缩模长项正在主导。")
        else:
            print("  [提示] 当前 ratio 已进入目标窗口，角度项正在主导。")

        if np.mean(cos_list) > 0.80:
            print("  [提示] cos 仍偏高，正交化还不够。")

    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    torch.save(ae.state_dict(), save_path)
    print("\n" + "=" * 180)
    print(f"[完成] 模型已保存至: {save_path}")


if __name__ == "__main__":
    run_two_stage_training()
