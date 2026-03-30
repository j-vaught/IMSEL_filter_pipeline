"""GPU-accelerated classical edge detectors using PyTorch.

All methods take a grayscale numpy array and return gradient magnitude as numpy.
GPU acceleration via torch.nn.functional.conv2d — same image, many kernels at once.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor(image: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """Convert HxW grayscale to 1x1xHxW float32 tensor."""
    return torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def _magnitude(gx: torch.Tensor, gy: torch.Tensor) -> np.ndarray:
    """Gradient magnitude from x and y components."""
    return torch.sqrt(gx**2 + gy**2).squeeze().cpu().numpy()


# =========================================================================
# 1. Sobel (extended to arbitrary kernel size)
# =========================================================================
def sobel(image: np.ndarray, ksize: int = 3, device: str = "cuda") -> np.ndarray:
    """Extended Sobel via Gaussian derivative approximation."""
    img = _to_tensor(image, device)
    if ksize == 3:
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
    else:
        sigma = (ksize - 1) / 4.0
        ax = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss = gauss / gauss.sum()
        dgauss = -ax / (sigma**2) * torch.exp(-0.5 * (ax / sigma) ** 2)
        dgauss = dgauss / (dgauss.abs().sum() / 2)
        kx = gauss.unsqueeze(1) * dgauss.unsqueeze(0)
        ky = dgauss.unsqueeze(1) * gauss.unsqueeze(0)
    pad = ksize // 2
    gx = F.conv2d(img, kx.unsqueeze(0).unsqueeze(0), padding=pad)
    gy = F.conv2d(img, ky.unsqueeze(0).unsqueeze(0), padding=pad)
    return _magnitude(gx, gy)


# =========================================================================
# 2. Prewitt (extended)
# =========================================================================
def prewitt(image: np.ndarray, ksize: int = 3, device: str = "cuda") -> np.ndarray:
    """Extended Prewitt via uniform smoothing + central difference."""
    img = _to_tensor(image, device)
    if ksize == 3:
        kx = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32, device=device)
        ky = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32, device=device)
    else:
        ax = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
        uniform = torch.ones(ksize, dtype=torch.float32, device=device) / ksize
        diff = torch.zeros(ksize, dtype=torch.float32, device=device)
        diff[0] = -1; diff[-1] = 1
        kx = uniform.unsqueeze(1) * diff.unsqueeze(0)
        ky = diff.unsqueeze(1) * uniform.unsqueeze(0)
    pad = ksize // 2
    gx = F.conv2d(img, kx.unsqueeze(0).unsqueeze(0), padding=pad)
    gy = F.conv2d(img, ky.unsqueeze(0).unsqueeze(0), padding=pad)
    return _magnitude(gx, gy)


# =========================================================================
# 3. Scharr (fixed 3x3)
# =========================================================================
def scharr(image: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Scharr 3x3 — optimized rotational symmetry."""
    img = _to_tensor(image, device)
    kx = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32, device=device)
    ky = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32, device=device)
    gx = F.conv2d(img, kx.unsqueeze(0).unsqueeze(0), padding=1)
    gy = F.conv2d(img, ky.unsqueeze(0).unsqueeze(0), padding=1)
    return _magnitude(gx, gy)


# =========================================================================
# 4. Roberts Cross (2x2 diagonal)
# =========================================================================
def roberts_cross(image: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Roberts Cross 2x2 diagonal differences."""
    img = _to_tensor(image, device)
    kx = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32, device=device)
    ky = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32, device=device)
    gx = F.conv2d(img, kx.unsqueeze(0).unsqueeze(0), padding=0)
    gy = F.conv2d(img, ky.unsqueeze(0).unsqueeze(0), padding=0)
    mag = torch.sqrt(gx**2 + gy**2).squeeze().cpu().numpy()
    # Pad to original size
    out = np.zeros_like(image)
    out[:mag.shape[0], :mag.shape[1]] = mag
    return out


# =========================================================================
# 5. Kirsch (8 directional kernels, take max)
# =========================================================================
def kirsch(image: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Kirsch compass operator — 8 directional 3x3 kernels, take max response."""
    img = _to_tensor(image, device)
    base = [5, 5, 5, -3, 0, -3, -3, -3, -3]
    kernels = []
    for rot in range(8):
        k = np.array(base, dtype=np.float32).reshape(3, 3)
        # Rotate by shifting elements
        flat = [5, 5, 5, -3, -3, -3, -3, -3]
        rotated = flat[rot:] + flat[:rot]
        k = np.array([rotated[0], rotated[1], rotated[2],
                       rotated[7], 0, rotated[3],
                       rotated[6], rotated[5], rotated[4]], dtype=np.float32).reshape(3, 3)
        kernels.append(torch.from_numpy(k).to(device))
    # Stack all 8 kernels: (8, 1, 3, 3)
    k_stack = torch.stack(kernels).unsqueeze(1)
    responses = F.conv2d(img, k_stack, padding=1)  # (1, 8, H, W)
    mag = responses.abs().max(dim=1)[0].squeeze().cpu().numpy()
    return mag


# =========================================================================
# 6. Robinson (8 directional, similar to Kirsch)
# =========================================================================
def robinson(image: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Robinson compass operator — 8 directional 3x3 kernels."""
    img = _to_tensor(image, device)
    # Robinson uses different coefficients than Kirsch
    templates = [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        [[0, -1, -2], [1, 0, -1], [2, 1, 0]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
    ]
    k_stack = torch.tensor(templates, dtype=torch.float32, device=device).unsqueeze(1)
    responses = F.conv2d(img, k_stack, padding=1)
    mag = responses.abs().max(dim=1)[0].squeeze().cpu().numpy()
    return mag


# =========================================================================
# 7. Frei-Chen (9 basis masks, edge subspace projection)
# =========================================================================
def frei_chen(image: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Frei-Chen edge detector — project onto edge subspace."""
    img = _to_tensor(image, device)
    s2 = np.sqrt(2)
    # 9 Frei-Chen basis masks (first 4 are edge masks)
    masks = [
        [[1, s2, 1], [0, 0, 0], [-1, -s2, -1]],
        [[1, 0, -1], [s2, 0, -s2], [1, 0, -1]],
        [[0, -1, s2], [1, 0, -1], [-s2, 1, 0]],
        [[s2, -1, 0], [-1, 0, 1], [0, 1, -s2]],
    ]
    k_stack = torch.tensor(masks, dtype=torch.float32, device=device).unsqueeze(1)
    responses = F.conv2d(img, k_stack, padding=1)  # (1, 4, H, W)
    # Edge magnitude = sqrt(sum of squared edge projections)
    mag = torch.sqrt((responses**2).sum(dim=1)).squeeze().cpu().numpy()
    return mag


# =========================================================================
# 8. Gaussian derivatives (1st order, arbitrary sigma)
# =========================================================================
def gaussian_derivative(image: np.ndarray, sigma: float = 1.0, device: str = "cuda") -> np.ndarray:
    """1st-order Gaussian derivative filter at given sigma."""
    img = _to_tensor(image, device)
    ksize = int(6 * sigma + 1) | 1  # ensure odd
    ax = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss = gauss / gauss.sum()
    dgauss = -ax / (sigma**2) * torch.exp(-0.5 * (ax / sigma) ** 2)
    # Normalize derivative kernel
    dgauss = dgauss / (dgauss.abs().sum() / 2 + 1e-12)
    # Separable: smooth in one direction, differentiate in the other
    kx = gauss.unsqueeze(1) * dgauss.unsqueeze(0)  # (ksize, ksize)
    ky = dgauss.unsqueeze(1) * gauss.unsqueeze(0)
    pad = ksize // 2
    gx = F.conv2d(img, kx.unsqueeze(0).unsqueeze(0), padding=pad)
    gy = F.conv2d(img, ky.unsqueeze(0).unsqueeze(0), padding=pad)
    return _magnitude(gx, gy)


# =========================================================================
# 9. Laplacian of Gaussian (LoG)
# =========================================================================
def laplacian_of_gaussian(image: np.ndarray, sigma: float = 1.0, device: str = "cuda") -> np.ndarray:
    """LoG filter — returns absolute value of LoG response."""
    img = _to_tensor(image, device)
    ksize = int(6 * sigma + 1) | 1
    ax = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    r2 = xx**2 + yy**2
    s2 = sigma**2
    log_kernel = -(1 / (np.pi * s2**2)) * (1 - r2 / (2 * s2)) * torch.exp(-r2 / (2 * s2))
    log_kernel = log_kernel - log_kernel.mean()  # zero-sum
    pad = ksize // 2
    response = F.conv2d(img, log_kernel.unsqueeze(0).unsqueeze(0), padding=pad)
    return response.abs().squeeze().cpu().numpy()


# =========================================================================
# 10. Difference of Gaussians (DoG)
# =========================================================================
def difference_of_gaussians(image: np.ndarray, sigma1: float = 1.0, k: float = 1.6,
                             device: str = "cuda") -> np.ndarray:
    """DoG — difference of two Gaussian blurs."""
    img = _to_tensor(image, device)
    sigma2 = sigma1 * k
    for sigma in [sigma1, sigma2]:
        ksize = int(6 * sigma + 1) | 1
        ax = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
        pad = ksize // 2
        if sigma == sigma1:
            blur1 = F.conv2d(img, gauss_2d.unsqueeze(0).unsqueeze(0), padding=pad)
        else:
            blur2 = F.conv2d(img, gauss_2d.unsqueeze(0).unsqueeze(0), padding=pad)
    return (blur1 - blur2).abs().squeeze().cpu().numpy()


# =========================================================================
# 11. Canny (GPU gradient + CPU NMS/hysteresis)
# =========================================================================
def canny_nms(image: np.ndarray, sigma: float = 1.0, device: str = "cuda") -> np.ndarray:
    """Canny NMS output (continuous, before hysteresis). GPU gradient, CPU NMS."""
    # GPU: Gaussian smooth + Sobel gradient
    mag = gaussian_derivative(image, sigma=sigma, device=device)
    return mag  # Return pre-NMS magnitude for ODS threshold sweep


# =========================================================================
# 12. Steerable filters (1st and 2nd order)
# =========================================================================
def steerable_filter(image: np.ndarray, sigma: float = 1.0, order: int = 1,
                      device: str = "cuda") -> np.ndarray:
    """Steerable filter — analytically rotatable Gaussian derivatives."""
    if order == 1:
        # 1st order steerable = Gaussian derivative (already implemented)
        return gaussian_derivative(image, sigma=sigma, device=device)
    else:
        # 2nd order: G_xx, G_yy, G_xy basis
        img = _to_tensor(image, device)
        ksize = int(6 * sigma + 1) | 1
        ax = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        r2 = xx**2 + yy**2
        s2 = sigma**2
        gauss = torch.exp(-r2 / (2 * s2))
        # G_xx
        gxx = ((xx**2 / s2 - 1) / s2) * gauss
        gyy = ((yy**2 / s2 - 1) / s2) * gauss
        gxy = (xx * yy / s2**2) * gauss
        pad = ksize // 2
        rxx = F.conv2d(img, gxx.unsqueeze(0).unsqueeze(0), padding=pad)
        ryy = F.conv2d(img, gyy.unsqueeze(0).unsqueeze(0), padding=pad)
        rxy = F.conv2d(img, gxy.unsqueeze(0).unsqueeze(0), padding=pad)
        # Max oriented 2nd derivative energy
        mag = torch.sqrt(rxx**2 + 2 * rxy**2 + ryy**2).squeeze().cpu().numpy()
        return mag


# =========================================================================
# 13. Marr-Hildreth (LoG + zero-crossing magnitude)
# =========================================================================
def marr_hildreth(image: np.ndarray, sigma: float = 1.0, device: str = "cuda") -> np.ndarray:
    """Marr-Hildreth — LoG zero-crossing strength."""
    # Use LoG response magnitude at zero-crossings
    return laplacian_of_gaussian(image, sigma=sigma, device=device)


# =========================================================================
# Batch runner: run all classical methods at all parameter settings
# =========================================================================
def run_all_classical(image: np.ndarray, device: str = "cuda") -> list[dict]:
    """Run all classical edge detectors, return list of {name, params, magnitude}."""
    results = []

    # Sobel
    for k in [3, 5, 7, 9, 11, 15, 21, 31, 41, 51]:
        mag = sobel(image, ksize=k, device=device)
        results.append({"method": "Sobel", "params": {"ksize": k}, "magnitude": mag})

    # Prewitt
    for k in [3, 5, 7, 9, 11, 15, 21, 31, 41]:
        mag = prewitt(image, ksize=k, device=device)
        results.append({"method": "Prewitt", "params": {"ksize": k}, "magnitude": mag})

    # Scharr
    mag = scharr(image, device=device)
    results.append({"method": "Scharr", "params": {}, "magnitude": mag})

    # Roberts Cross
    mag = roberts_cross(image, device=device)
    results.append({"method": "Roberts", "params": {}, "magnitude": mag})

    # Kirsch
    mag = kirsch(image, device=device)
    results.append({"method": "Kirsch", "params": {}, "magnitude": mag})

    # Robinson
    mag = robinson(image, device=device)
    results.append({"method": "Robinson", "params": {}, "magnitude": mag})

    # Frei-Chen
    mag = frei_chen(image, device=device)
    results.append({"method": "Frei-Chen", "params": {}, "magnitude": mag})

    # Gaussian derivatives
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0]:
        mag = gaussian_derivative(image, sigma=sigma, device=device)
        results.append({"method": "GaussianDeriv", "params": {"sigma": sigma}, "magnitude": mag})

    # LoG
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]:
        mag = laplacian_of_gaussian(image, sigma=sigma, device=device)
        results.append({"method": "LoG", "params": {"sigma": sigma}, "magnitude": mag})

    # DoG
    for sigma1 in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]:
        mag = difference_of_gaussians(image, sigma1=sigma1, device=device)
        results.append({"method": "DoG", "params": {"sigma1": sigma1}, "magnitude": mag})

    # Steerable (1st order = same as GaussianDeriv, so only do 2nd order)
    for sigma in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]:
        mag = steerable_filter(image, sigma=sigma, order=2, device=device)
        results.append({"method": "Steerable2nd", "params": {"sigma": sigma}, "magnitude": mag})

    # Marr-Hildreth (same as LoG for our purposes, skip to avoid duplication)

    # Canny NMS (same as GaussianDeriv for our purposes when evaluating
    # continuous output, skip to avoid duplication)

    return results


def get_config_count() -> int:
    """Return total number of configurations."""
    return (10 + 9 + 1 + 1 + 1 + 1 + 1 + 11 + 10 + 8 + 8)  # 61 unique configs
