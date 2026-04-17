import os
import copy
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T
import numpy as np


# ============================================================
# I-BAU defense evaluation on the NEW dual-trigger model
# - AE(A) path updated
# - final coupled model path updated
# - keeps BA / ASR_A / ASR_B / ASR_AB evaluation
# - keeps author-style fixed-point outer update
# ============================================================


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Model: ResNet-18 for CIFAR-10
# -----------------------------
def build_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# -----------------------------
# CIFAR-10 normalization
# -----------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
EPS = 1e-8


def get_mean_std(device: torch.device):
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


# -----------------------------
# AE / Trigger components
# -----------------------------
class PoisonAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def generate_block_shift_matrix(H, W, blocks_h, blocks_w, pattern, device):
    shift_matrix = torch.zeros((H, W), device=device)
    h_step, w_step = H // blocks_h, W // blocks_w
    for i in range(blocks_h):
        for j in range(blocks_w):
            h_start = i * h_step
            h_end = (i + 1) * h_step if i < blocks_h - 1 else H
            w_start = j * w_step
            w_end = (j + 1) * w_step if j < blocks_w - 1 else W
            shift_matrix[h_start:h_end, w_start:w_end] = pattern[i, j]
    return shift_matrix


class FrequencyPhaseJitterTransform:
    def __init__(self, r_min=0.6875, r_max=0.8750, h=32, w=32, n=0.5):
        self.r_min = r_min
        self.r_max = r_max
        self.h = h
        self.w = w
        self.n = n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        device_x = x.device
        pattern = torch.tensor(
            [[self.n * math.pi, -self.n * math.pi],
             [-self.n * math.pi, self.n * math.pi]],
            device=device_x,
            dtype=x.dtype,
        )
        jitter_map = generate_block_shift_matrix(self.h, self.w, 2, 2, pattern, device_x)

        X = torch.fft.fft2(x)
        mag, phase = torch.abs(X), torch.angle(X)

        fy = torch.fft.fftfreq(x.shape[-2], d=1.0, device=device_x)
        fx = torch.fft.fftfreq(x.shape[-1], d=1.0, device=device_x)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(xx**2 + yy**2)
        r_norm = r / (r.max() + 1e-6)

        mask = (r_norm >= self.r_min) & (r_norm < self.r_max)
        phase_new = torch.where(mask, phase + jitter_map, phase)

        out = torch.fft.ifft2(mag * torch.exp(1j * phase_new)).real
        return out.clamp(0.0, 1.0)


def get_freq_mask_A(H, W, r_low=0.1, r_high=0.5, device_target="cpu"):
    fy = torch.fft.fftfreq(H, d=1.0, device=device_target)
    fx = torch.fft.fftfreq(W, d=1.0, device=device_target)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    r_norm = r / (r.max() + 1e-6)
    return (r_norm >= r_low) & (r_norm <= r_high)


def build_A_with_budget(ae: nn.Module, imgs: torch.Tensor, mask_A: torch.Tensor, target_mse: float) -> torch.Tensor:
    gen_A_raw = ae(imgs)
    diff = gen_A_raw - imgs

    fft_diff_filtered = torch.fft.fft2(diff) * mask_A.unsqueeze(0).unsqueeze(1)
    delta = torch.real(torch.fft.ifft2(fft_diff_filtered))

    delta_mse = torch.mean(delta ** 2, dim=(1, 2, 3), keepdim=True)
    alpha = torch.clamp(torch.sqrt(target_mse / (delta_mse + EPS)), 0.0, 1.0)

    imgs_A = torch.clamp(imgs + delta * alpha, 0.0, 1.0)
    return imgs_A


class TriggerSuite:
    def __init__(self, ae_model_path: str, target_psnr: float, device: torch.device):
        self.device = device
        self.target_psnr = target_psnr
        self.target_mse = 10 ** (-target_psnr / 10.0)

        self.ae = PoisonAE().to(device)
        sd = torch.load(ae_model_path, map_location=device)
        self.ae.load_state_dict(sd)
        self.ae.eval()

        self.mask_A = get_freq_mask_A(32, 32, 0.1, 0.5, device_target=device)
        self.trigger_B = FrequencyPhaseJitterTransform(n=0.5)

        self.mean, self.std = get_mean_std(device)

    @torch.no_grad()
    def apply_A(self, x_norm: torch.Tensor) -> torch.Tensor:
        x = denormalize(x_norm, self.mean, self.std).clamp(0.0, 1.0)
        x = build_A_with_budget(self.ae, x, self.mask_A, self.target_mse)
        return normalize(x, self.mean, self.std)

    @torch.no_grad()
    def apply_B(self, x_norm: torch.Tensor) -> torch.Tensor:
        x = denormalize(x_norm, self.mean, self.std).clamp(0.0, 1.0)
        x = self.trigger_B(x)
        return normalize(x, self.mean, self.std)

    @torch.no_grad()
    def apply_AB(self, x_norm: torch.Tensor) -> torch.Tensor:
        x = self.apply_A(x_norm)
        x = self.apply_B(x)
        return x


# -----------------------------
# Dataset wrapper
# -----------------------------
class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return x, y, idx


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    seed: int = 2026
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # >>> NEW PATHS <<<
    model_path: str = "/content/final_coupled_resnet18_softgate_psnr38_badnet_ref_ratio45_50_statfixed_100.pth"
    ae_model_path: str = "/content/ae_A_softgate_psnr38_badnet_ref_ratio200%_statfixed (1).pth"
    save_dir: str = "/content/ibau_softgate_psnr38_eval"

    target_label: int = 0
    num_classes: int = 10
    target_psnr: float = 38.0

    clean_subset_size: int = 5000
    batch_size_unlearn: int = 100
    test_batch_size: int = 256
    num_workers: int = 2

    # I-BAU outer loop
    n_rounds: int = 50
    K: int = 5
    outer_optim: str = "Adam"
    outer_lr: float = 1e-3

    # perturbation optimization
    inner_pert_lr: float = 10.0
    fixed_point_inner_lr: float = 0.1
    pert_reg: float = 1e-3
    patch_portion: float = 0.01

    # safety / monitoring
    grad_clip: float = 1.0
    save_every: int = 1

    data_root: str = "/content/data"
    download: bool = True


# -----------------------------
# Data
# -----------------------------
def build_dataloaders(cfg: Config):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=cfg.data_root,
        train=True,
        download=cfg.download,
        transform=transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=cfg.data_root,
        train=False,
        download=cfg.download,
        transform=transform,
    )

    indexed_train = IndexedDataset(train_set)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    perm = torch.randperm(len(indexed_train), generator=g).tolist()
    clean_indices = perm[:cfg.clean_subset_size]
    clean_subset = Subset(indexed_train, clean_indices)

    clean_loader = DataLoader(
        clean_subset,
        batch_size=cfg.batch_size_unlearn,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return clean_loader, test_loader


# -----------------------------
# Model load
# -----------------------------
def load_model(cfg: Config, device: torch.device) -> nn.Module:
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {cfg.model_path}")

    model = build_resnet18_cifar10(num_classes=cfg.num_classes)
    ckpt = torch.load(cfg.model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f">>> Missing keys: {missing}")
    if unexpected:
        print(f">>> Unexpected keys: {unexpected}")

    model.to(device)
    return model


# -----------------------------
# Hypergradient utilities
# -----------------------------
class DifferentiableOptimizer:
    def __init__(self, loss_f, dim_mult):
        self.loss_f = loss_f
        self.dim_mult = dim_mult
        self.curr_loss = None

    def step(self, params, hparams, create_graph):
        raise NotImplementedError

    def __call__(self, params, hparams, create_graph=True):
        with torch.enable_grad():
            return self.step(params, hparams, create_graph)

    def get_loss(self, params, hparams):
        self.curr_loss = self.loss_f(params, hparams)
        return self.curr_loss


class GradientDescent(DifferentiableOptimizer):
    def __init__(self, loss_f, step_size):
        super().__init__(loss_f, dim_mult=1)
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size

    def step(self, params, hparams, create_graph):
        loss = self.get_loss(params, hparams)
        sz = self.step_size_f(hparams)
        grads = torch.autograd.grad(loss, params, create_graph=create_graph)
        return [w - sz * g for w, g in zip(params, grads)]


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(
        output,
        inputs,
        grad_outputs=grad_outputs,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)
    return grad_outer_w, grad_outer_hparams


def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.reshape([-1]) for xx in list_tx])


def fixed_point(
    params: List[torch.Tensor],
    hparams: List[torch.Tensor],
    K: int,
    fp_map: Callable[[List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]],
    outer_loss: Callable[[List[torch.Tensor], List[torch.Tensor]], torch.Tensor],
    tol=1e-10,
    set_grad=True,
    stochastic=False,
) -> List[torch.Tensor]:
    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    if not stochastic:
        w_mapped = fp_map(params, hparams)

    vs = [torch.zeros_like(w) for w in params]
    vs_vec = cat_list_to_tensor(vs)

    for _ in range(K):
        vs_prev_vec = vs_vec
        if stochastic:
            w_mapped = fp_map(params, hparams)
            vs = torch.autograd.grad(w_mapped, params, grad_outputs=vs, retain_graph=False)
        else:
            vs = torch.autograd.grad(w_mapped, params, grad_outputs=vs, retain_graph=True)

        vs = [v + gow for v, gow in zip(vs, grad_outer_w)]
        vs_vec = cat_list_to_tensor(vs)

        if float(torch.norm(vs_vec - vs_prev_vec)) < tol:
            break

    if stochastic:
        w_mapped = fp_map(params, hparams)

    grads = torch.autograd.grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
    grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_clean(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_asr(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_label: int,
    trigger_fn: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    model.eval()
    success, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        mask = (y != target_label)
        if mask.sum().item() == 0:
            continue

        x = x[mask]
        y = y[mask]

        x_t = trigger_fn(x)
        pred = model(x_t).argmax(dim=1)

        success += (pred == target_label).sum().item()
        total += y.size(0)

    return success / max(total, 1)


@torch.no_grad()
def evaluate_target_ratio(model: nn.Module, loader: DataLoader, device: torch.device, target_label: int) -> float:
    model.eval()
    total, target_pred = 0, 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        target_pred += (pred == target_label).sum().item()
        total += x.size(0)
    return target_pred / max(total, 1)


def evaluate_four_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    trigger_suite: TriggerSuite,
) -> Dict[str, float]:
    ba = evaluate_clean(model, loader, device)
    asr_a = evaluate_asr(model, loader, device, cfg.target_label, trigger_suite.apply_A)
    asr_b = evaluate_asr(model, loader, device, cfg.target_label, trigger_suite.apply_B)
    asr_ab = evaluate_asr(model, loader, device, cfg.target_label, trigger_suite.apply_AB)
    tpr = evaluate_target_ratio(model, loader, device, cfg.target_label)

    return {
        "BA": ba,
        "ASR_A": asr_a,
        "ASR_B": asr_b,
        "ASR_AB": asr_ab,
        "TargetPredRatio": tpr,
    }


# -----------------------------
# Main I-BAU loop
# -----------------------------
def ibau_unlearn(
    model: nn.Module,
    clean_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    trigger_suite: TriggerSuite,
):
    os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.outer_optim == "SGD":
        outer_opt = torch.optim.SGD(model.parameters(), lr=cfg.outer_lr)
    else:
        outer_opt = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    history: List[Dict[str, float]] = []

    images_list, labels_list = [], []
    for images, labels, _ in clean_loader:
        images_list.append(images)
        labels_list.append(labels)

    metrics0 = evaluate_four_metrics(model, test_loader, device, cfg, trigger_suite)
    print(
        f"Round 00 | BA={metrics0['BA']:.4f} | ASR_A={metrics0['ASR_A']:.4f} | "
        f"ASR_B={metrics0['ASR_B']:.4f} | ASR_AB={metrics0['ASR_AB']:.4f} | "
        f"TargetPredRatio={metrics0['TargetPredRatio']:.4f}"
    )
    history.append({"round": 0, **metrics0})

    def loss_inner(perturb, model_params):
        images = images_list[0].to(device)
        labels = labels_list[0].long().to(device)
        per_img = images + perturb[0]
        per_logits = model(per_img)
        loss = F.cross_entropy(per_logits, labels, reduction="none")
        loss_regu = torch.mean(-loss) + cfg.pert_reg * torch.pow(torch.norm(perturb[0]), 2)
        return loss_regu

    inner_opt = GradientDescent(loss_inner, cfg.fixed_point_inner_lr)

    for round_idx in range(1, cfg.n_rounds + 1):
        batch_pert = torch.zeros((1, 3, 32, 32), requires_grad=True, device=device)
        batch_opt = torch.optim.SGD(params=[batch_pert], lr=cfg.inner_pert_lr)

        # author-style perturbation optimization
        for images, labels, _ in clean_loader:
            images = images.to(device)
            ori_lab = torch.argmax(model(images), dim=1).long()

            per_logits = model(images + batch_pert)
            loss = F.cross_entropy(per_logits, ori_lab, reduction="mean")
            loss_regu = torch.mean(-loss) + cfg.pert_reg * torch.pow(torch.norm(batch_pert), 2)

            batch_opt.zero_grad()
            loss_regu.backward(retain_graph=True)
            batch_opt.step()

        pert = batch_pert.detach()

        for batchnum in range(len(images_list)):
            def loss_outer(perturb, model_params):
                images = images_list[batchnum].to(device)
                labels = labels_list[batchnum].long().to(device)

                patching = torch.zeros_like(images, device=device)
                number = images.shape[0]
                num_patch = max(1, int(number * cfg.patch_portion))
                rand_idx = random.sample(list(np.arange(number)), num_patch)
                patching[rand_idx] = perturb[0]

                unlearn_imgs = images + patching
                logits = model(unlearn_imgs)
                return F.cross_entropy(logits, labels)

            outer_opt.zero_grad()
            fixed_point([pert], list(model.parameters()), cfg.K, inner_opt, loss_outer)

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            outer_opt.step()

        metrics = evaluate_four_metrics(model, test_loader, device, cfg, trigger_suite)
        print(
            f"Round {round_idx:02d} | BA={metrics['BA']:.4f} | ASR_A={metrics['ASR_A']:.4f} | "
            f"ASR_B={metrics['ASR_B']:.4f} | ASR_AB={metrics['ASR_AB']:.4f} | "
            f"TargetPredRatio={metrics['TargetPredRatio']:.4f} | mode=author_style_fixed_point"
        )
        history.append({"round": round_idx, **metrics})

        if metrics["BA"] <= 0.15 and metrics["TargetPredRatio"] >= 0.90:
            print(">>> Early stop: target-collapse detected.")
            break

        if round_idx % cfg.save_every == 0 or round_idx == cfg.n_rounds:
            save_path = os.path.join(cfg.save_dir, f"ibau_round_{round_idx}.pth")
            torch.save(model.state_dict(), save_path)
            print(f">>> Saved checkpoint: {save_path}")

    final_path = os.path.join(cfg.save_dir, "ibau_final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f">>> Final model saved to: {final_path}")

    log_path = os.path.join(cfg.save_dir, "ibau_history.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        for row in history:
            f.write(f"{row}\n")
    print(f">>> History saved to: {log_path}")

    return history


# -----------------------------
# Entry
# -----------------------------
def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    print(f">>> Device: {device}")
    print(f">>> Loading NEW coupled model from: {cfg.model_path}")
    print(f">>> Loading NEW AE(A) from: {cfg.ae_model_path}")
    print(f">>> Defense mode: author-style fixed-point I-BAU")
    print(f">>> Save dir: {cfg.save_dir}")

    clean_loader, test_loader = build_dataloaders(cfg)
    model = load_model(cfg, device)
    trigger_suite = TriggerSuite(cfg.ae_model_path, cfg.target_psnr, device)

    ibau_unlearn(model, clean_loader, test_loader, device, cfg, trigger_suite)


if __name__ == "__main__":
    main()
