import math
import pdb

import torch
import torch.nn.functional as F

# --------------------------------------
# pooling
# --------------------------------------


def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_max_pool2d(x, (1,1)) # alternative


def spoc(x):
    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_avg_pool2d(x, (1,1)) # alternative


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


def rmac(x, L=3, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    (tmp, idx) = torch.min(
        torch.abs(((w**2 - w * b) / w**2) - ovr), 0
    )  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = (
            torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2
        )  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = (
            torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2
        )  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v


def roipool(x, rpool, L=3, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    _, idx = torch.min(
        torch.abs(((w**2 - w * b) / w**2) - ovr), 0
    )  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    vecs = []
    vecs.append(rpool(x).unsqueeze(1))

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = (
            torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b).int() - wl2
        )  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = (
            torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b).int() - wl2
        )  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                vecs.append(rpool(x.narrow(2, i_, wl).narrow(3, j_, wl)).unsqueeze(1))

    return torch.cat(vecs, dim=1)


# --------------------------------------
# normalization
# --------------------------------------


def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


def powerlaw(x, eps=1e-6):
    x = x + eps
    return x.abs().sqrt().mul(x.sign())


# --------------------------------------
# Bayesian loss functions
# --------------------------------------


def negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=0.3):
    """
    Compute the negative log-likelihood for Bayesian triplet loss
    """
    # Ensure variances are positive and not too small
    eps = 1e-8
    varA = torch.clamp(varA, min=eps)
    varP = torch.clamp(varP, min=eps)
    varN = torch.clamp(varN, min=eps)
    
    dist_ap = torch.sum((muA - muP) ** 2, dim=0)
    dist_an = torch.sum((muA - muN) ** 2, dim=0)
    
    # Add variance terms
    var_ap = torch.sum(varA + varP, dim=0)
    var_an = torch.sum(varA + varN, dim=0)
    
    # Ensure variance terms are positive
    var_ap = torch.clamp(var_ap, min=eps)
    var_an = torch.clamp(var_an, min=eps)
    
    # Compute log-likelihood with numerical stability
    log_likelihood = -0.5 * (dist_ap / var_ap + torch.log(var_ap) + 
                            dist_an / var_an + torch.log(var_an))
    
    # Add margin term
    loss = torch.clamp(dist_ap - dist_an + margin, min=0.0)
    
    total_loss = -log_likelihood + loss
    
    # Check for NaN and replace with finite values
    if torch.isnan(total_loss).any():
        print(f"Warning: NaN detected in loss. dist_ap: {dist_ap}, dist_an: {dist_an}, var_ap: {var_ap}, var_an: {var_an}")
        total_loss = torch.where(torch.isnan(total_loss), torch.tensor(0.0, device=total_loss.device), total_loss)
    
    return total_loss.mean()


def kl_div_gauss(mu1, var1, mu2, var2):
    """
    Compute KL divergence between two Gaussian distributions
    """
    eps = 1e-8
    var1 = torch.clamp(var1, min=eps)
    var2 = torch.clamp(var2, min=eps)
    
    kl = 0.5 * torch.sum(var1 / var2 + (mu1 - mu2) ** 2 / var2 - 1 - torch.log(var1 / var2))
    
    # Check for NaN and replace with finite values
    if torch.isnan(kl):
        print(f"Warning: NaN detected in KL divergence. var1: {var1}, var2: {var2}")
        kl = torch.tensor(0.0, device=kl.device)
    
    return kl


def kl_div_vMF(mu, var):
    """
    Compute KL divergence for von Mises-Fisher distribution
    """
    # For vMF, we assume mu is normalized and var is the concentration parameter
    eps = 1e-8
    kappa = torch.clamp(var, min=eps)
    d = mu.size(0)
    
    # Compute Bessel function ratio with numerical stability
    bessel_ratio = torch.sqrt(1 + 4 * kappa ** 2) / (2 * kappa)
    
    # Compute KL divergence
    two_pi = torch.tensor(2 * math.pi, device=kappa.device)
    kl = kappa * bessel_ratio + (d/2 - 1) * torch.log(kappa) - \
         (d/2) * torch.log(two_pi) - torch.log(bessel_ratio)
    
    # Check for NaN and replace with finite values
    if torch.isnan(kl):
        print(f"Warning: NaN detected in vMF KL divergence. kappa: {kappa}")
        kl = torch.tensor(0.0, device=kl.device)
    
    return kl
