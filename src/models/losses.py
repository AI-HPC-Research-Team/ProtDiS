import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch_npu

def orthogonal_loss(f_super, f_ss):
    """
    计算两个特征向量之间的正交损失。
    参数:
        f_super: Tensor，形状 (batch, feature_dim)
        f_ss: Tensor，形状 (batch, feature_dim)
    返回:
        平均正交损失：内积的平方的平均值
    """
    # 对每个样本归一化
    f_super_norm = F.normalize(f_super, p=2, dim=-1)
    f_ss_norm = F.normalize(f_ss, p=2, dim=-1)
    # 计算每个样本的内积
    dot_product = (f_super_norm * f_ss_norm).sum(dim=-1)
    # 平方后取均值
    loss = torch.mean(dot_product ** 2)
    return loss

def info_nce_loss(z1: torch.Tensor,
                  z2: torch.Tensor,
                  temperature: float) -> torch.Tensor:
    """
    Compute InfoNCE loss between z1 and z2.
    z1, z2: (batch, dim)
    """
    batch_size = z1.size(0)
    # L2 normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # 计算相似度矩阵
    logits = torch.matmul(z1, z2.t()) / temperature  # (B, B)
    labels = torch.arange(batch_size, device=z1.device)
    # cross_entropy: -E[log p_i!=j]
    loss = F.cross_entropy(logits, labels)
    return loss

def redundancy_reduction_loss(z_list, lamb_var=1.0, lamb_cov=1.0, eps=1e-4):
    """
    z_list: List[Tensor], 每个形状 [N, D]
    返回：标量损失。含两部分：
      1) variance term：每个编码的每维方差 ~ 1
      2) cross-correlation decorrelation：不同编码之间的相关矩阵接近对角/零
    """
    N = z_list[0].shape[0]
    K = len(z_list)

    # 标准化（去均值、按维度标准差缩放）
    zs = []
    var_loss = 0.0
    for z in z_list:
        zc = z - z.mean(dim=0, keepdim=True)
        std = zc.std(dim=0, unbiased=False) + eps
        zn = zc / std
        zs.append(zn)
        # 让方差趋近 1（VICReg 的 variance term：鼓励std>=1；这里简单做 (std-1)^2）
        var_loss = var_loss + ((std - 1.0)**2).mean()

    # 跨编码的交相关矩阵（按批）
    # 对所有 pair 求相关矩阵并惩罚 off-diagonal
    cov_loss = 0.0
    num_pairs = 0
    for i in range(K):
        for j in range(i+1, K):
            # 相关矩阵：C = (Zi^T Zj) / N
            C = (zs[i].T @ zs[j]) / N  # [D, D]
            # 目标：C ~ 0 （全部元素接近 0）
            # 你也可以只惩罚 off-diagonal，这里直接全惩罚更简单
            cov_loss = cov_loss + (C**2).mean()
            num_pairs += 1
    if num_pairs > 0:
        cov_loss = cov_loss / num_pairs

    return lamb_var * var_loss + lamb_cov * cov_loss

def hsic_loss(z_list, sigma=5.0):
    """
    z_list: List[Tensor], 每个形状 [N, D]
    """
    N = z_list[0].shape[0]
    K = len(z_list)

    def rbf(x, sigma):
        xx = (x**2).sum(dim=1, keepdim=True)
        xy = x @ x.t()
        dist = xx + xx.t() - 2*xy
        return torch.exp(-dist / (2*sigma**2))

    def hsic(x, y, sigma):
        K = rbf(x, sigma)
        L = rbf(y, sigma)
        H = torch.eye(K.size(0), device=x.device) - 1.0 / K.size(0)
        KH = K @ H
        LH = L @ H
        return torch.trace(KH @ LH) / (K.size(0) - 1)**2

    loss = 0.0
    num_pairs = 0
    for i in range(K):
        for j in range(i+1, K):
            loss = loss + hsic(z_list[i], z_list[j], sigma)
            num_pairs += 1
    if num_pairs > 0:
        loss = loss / num_pairs

    return loss

def dcc_loss(z_list):
    """
    z_list: List[Tensor], 每个形状 [N, D]
    """
    N = z_list[0].shape[0]
    K = len(z_list)

    def pdist(a):
        r = (a**2).sum(dim=1, keepdim=True)
        return r + r.t() - 2 * (a @ a.t())

    def dcc(x, y):
        A = pdist(x).sqrt()
        B = pdist(y).sqrt()

        A -= A.mean(dim=0, keepdim=True)
        A -= A.mean(dim=1, keepdim=True)
        B -= B.mean(dim=0, keepdim=True)
        B -= B.mean(dim=1, keepdim=True)

        dcov = (A * B).mean()
        dvar_x = (A * A).mean().sqrt()
        dvar_y = (B * B).mean().sqrt()
        return dcov / (dvar_x * dvar_y + 1e-8)

    dcc_loss = 0.0
    num_pairs = 0
    for i in range(K):
        for j in range(i+1, K):
            dcc_loss = dcc_loss + dcc(z_list[i], z_list[j])
            num_pairs += 1
    if num_pairs > 0:
        dcc_loss = dcc_loss / num_pairs

    return dcc_loss
