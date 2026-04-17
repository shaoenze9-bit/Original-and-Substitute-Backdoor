import os
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

# ==========================================
# 1. 基础配置
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AE_PATH = "/content/A_WarmStart_TwoStage_StableRotate/ae_A_softgate_psnr38_badnet_ref_ratio45_50_statfixed.pth"

TARGET_PSNR = 38.0
TARGET_MSE = 10 ** (-TARGET_PSNR / 10.0)
EPS = 1e-8

# ==========================================
# 2. 模型与变换组件
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

# ==========================================
# 3. 与本次训练保持一致的 A 构造方式
# ==========================================
def build_mask_A(device, h=32, w=32, r_min=0.6875, r_max=0.8750):
    fy = torch.fft.fftfreq(h, d=1.0, device=device)
    fx = torch.fft.fftfreq(w, d=1.0, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    r_norm = r / (r.max() + 1e-6)
    mask_A = (r_norm >= r_min) & (r_norm <= r_max)
    return mask_A


def build_A_with_budget(ae, imgs, mask_A):
    """
    完全按本次训练定义构造 A：
    1) AE 生成 raw A
    2) 转成差分 diff
    3) 只保留 mask_A 频段
    4) 缩放到 TARGET_PSNR 对应预算
    """
    gen_A_raw = ae(imgs)  # [B, 3, H, W]

    diff = gen_A_raw - imgs
    fft_diff = torch.fft.fft2(diff)
    fft_diff_filtered = fft_diff * mask_A.unsqueeze(0).unsqueeze(1)
    delta = torch.real(torch.fft.ifft2(fft_diff_filtered))

    delta_mse = torch.mean(delta ** 2, dim=(1, 2, 3), keepdim=True)
    alpha = torch.clamp(torch.sqrt(TARGET_MSE / (delta_mse + EPS)), 0.0, 1.0)
    delta_scaled = delta * alpha

    imgs_A = torch.clamp(imgs + delta_scaled, 0.0, 1.0)
    mseA_each = torch.mean((imgs_A - imgs) ** 2, dim=(1, 2, 3))

    return imgs_A, delta_scaled, alpha, mseA_each


# ==========================================
# 4. 指标函数
# ==========================================
def calc_psnr_ssim(clean_np, poison_np):
    p = psnr_func(clean_np, poison_np, data_range=1.0)
    s = ssim_func(clean_np, poison_np, data_range=1.0, channel_axis=2)
    return p, s


# ==========================================
# 5. 全量评估：测真正参与训练/测试定义的 A / B / AB 的隐蔽性
# ==========================================
def run_full_dataset_evaluation():
    ae = PoisonAE().to(DEVICE)
    if os.path.exists(AE_PATH):
        ae.load_state_dict(torch.load(AE_PATH, map_location=DEVICE))
        print(f">>> 成功加载 AE 模型: {AE_PATH}")
    else:
        print(f"!!! 错误: 路径不存在 {AE_PATH}")
        return
    ae.eval()

    transform = T.ToTensor()
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    full_ds = ConcatDataset([train_ds, test_ds])
    loader = DataLoader(full_ds, batch_size=1, shuffle=True)

    mask_A = build_mask_A(DEVICE, h=32, w=32, r_min=0.6875, r_max=0.8750)

    stats = {
        "A":  {"p": [], "s": [], "mse": []},
        "B":  {"p": [], "s": [], "mse": []},
        "AB": {"p": [], "s": [], "mse": []},
    }

    example_data = None

    print(f">>> 开始评估全量数据 (60,000张)，设备: {DEVICE}")
    print(f">>> 当前评估对象：本次实验 AE 生成的 A（目标 PSNR = {TARGET_PSNR} dB）")
    print(">>> A 的定义：AE raw 输出 -> diff -> 频域筛选 -> budget 缩放")
    print(">>> B 的定义：BadNetTrigger")
    print(">>> AB 的定义：先构造训练态 A，再叠加 B")
    print(">>> 当前实验：softgate_psnr38_badnet_ref_ratio45_50_statfixed")

    with torch.no_grad():
        for i, (img, _) in enumerate(loader):
            img = img.to(DEVICE)

            # 真正训练时使用的 A
            img_A, delta_scaled, alpha, mseA_each = build_A_with_budget(ae, img, mask_A)

            # B
            img_B = trigger_B(img)

            # AB：先训练态 A，再加 B
            img_AB = trigger_B(img_A)

            cln_np = img.squeeze(0).cpu().permute(1, 2, 0).numpy()
            a_np   = img_A.squeeze(0).cpu().permute(1, 2, 0).numpy()
            b_np   = img_B.squeeze(0).cpu().permute(1, 2, 0).numpy()
            ab_np  = img_AB.squeeze(0).cpu().permute(1, 2, 0).numpy()

            for key, target_np in zip(["A", "B", "AB"], [a_np, b_np, ab_np]):
                p, s = calc_psnr_ssim(cln_np, target_np)
                mse = np.mean((cln_np - target_np) ** 2)

                stats[key]["p"].append(p)
                stats[key]["s"].append(s)
                stats[key]["mse"].append(mse)

            if i == 0:
                example_data = {
                    "imgs": [cln_np, a_np, b_np, ab_np],
                    "metrics": [
                        (stats["A"]["p"][-1],  stats["A"]["s"][-1]),
                        (stats["B"]["p"][-1],  stats["B"]["s"][-1]),
                        (stats["AB"]["p"][-1], stats["AB"]["s"][-1]),
                    ],
                    "alpha": alpha.item(),
                    "mseA": mseA_each.item(),
                }

            if (i + 1) % 5000 == 0:
                print(
                    f"进度 {i+1:5d}/60000 | "
                    f"A-PSNR: {np.mean(stats['A']['p']):.4f} | "
                    f"B-PSNR: {np.mean(stats['B']['p']):.4f} | "
                    f"AB-PSNR: {np.mean(stats['AB']['p']):.4f}"
                )

    # ==========================================
    # 最终汇总
    # ==========================================
    print("\n" + "=" * 78)
    print(f"{'投毒策略':<12} | {'平均 PSNR (dB)':<18} | {'平均 SSIM':<12} | {'平均 MSE':<12}")
    print("-" * 78)
    for k in ["A", "B", "AB"]:
        print(
            f"{k:<12} | "
            f"{np.mean(stats[k]['p']):<18.4f} | "
            f"{np.mean(stats[k]['s']):<12.6f} | "
            f"{np.mean(stats[k]['mse']):<12.8f}"
        )
    print("=" * 78)

    print("\n>>> 第一张样本的训练态 A 诊断信息：")
    print(f"alpha = {example_data['alpha']:.6f}")
    print(f"MSE(A vs clean) = {example_data['mseA']:.8f}")
    print(f"PSNR(A vs clean) = {example_data['metrics'][0][0]:.4f}")
    print(f"SSIM(A vs clean) = {example_data['metrics'][0][1]:.6f}")

    # ==========================================
    # 可视化
    # ==========================================
    titles = ["Clean", f"Poison A ({TARGET_PSNR}dB budget)", "Poison B", "Coupled AB"]
    plt.figure(figsize=(18, 9))

    for j in range(4):
        plt.subplot(2, 4, j + 1)
        plt.imshow(example_data["imgs"][j])
        plt.title(titles[j])
        if j > 0:
            p_val, s_val = example_data["metrics"][j - 1]
            plt.xlabel(f"PSNR: {p_val:.2f}\nSSIM: {s_val:.4f}")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 4, j + 5)
        if j == 0:
            res = np.ones((32, 32, 3)) * 0.5
        else:
            res = np.clip((example_data["imgs"][j] - example_data["imgs"][0]) * 5 + 0.5, 0, 1)
        plt.imshow(res)
        plt.title(f"Residual {titles[j]} (x5)")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_full_dataset_evaluation()
