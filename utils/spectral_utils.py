#
# Spectral-GS utilities: DCT-based spectral entropy loss and densification guidance.
# Reference: "Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy"
#

import torch
import torch.nn.functional as F


def _dct_1d(x: torch.Tensor) -> torch.Tensor:
    """1D DCT-II along the last dimension (differentiable via torch.fft)."""
    N = x.shape[-1]
    shape = x.shape
    x = x.contiguous().view(-1, N)

    # Reorder: interleave even and odd indices
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # FFT on reordered signal
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))  # (..., N, 2)

    # Phase shift: exp(-j * pi * k / (2N))
    k = torch.arange(N, dtype=x.dtype, device=x.device).unsqueeze(0)
    W_r = torch.cos(-torch.pi * k / (2 * N))
    W_i = torch.sin(-torch.pi * k / (2 * N))

    V = Vc[:, :N, 0] * W_r - Vc[:, :N, 1] * W_i  # real part of phase-shifted FFT

    return (2 * V).view(*shape)


def dct_2d(image: torch.Tensor) -> torch.Tensor:
    """2D DCT-II of image tensor (..., H, W). Differentiable."""
    # Apply 1D DCT along width (last dim)
    X1 = _dct_1d(image)
    # Apply 1D DCT along height (second-to-last dim)
    X2 = _dct_1d(X1.transpose(-1, -2)).transpose(-1, -2)
    return X2


def spectral_entropy(image: torch.Tensor) -> torch.Tensor:
    """Compute scalar spectral entropy H = -sum(p * log(p)) of an image.

    Args:
        image: (C, H, W) or (H, W) tensor

    Returns:
        Scalar tensor representing mean spectral entropy across channels.
    """
    dct = dct_2d(image)
    power = dct ** 2  # power spectrum

    if power.dim() == 3:
        power_flat = power.flatten(1)      # (C, H*W)
    else:
        power_flat = power.flatten(0).unsqueeze(0)  # (1, H*W)

    p = power_flat / (power_flat.sum(dim=-1, keepdim=True) + 1e-10)
    entropy = -(p * torch.log(p + 1e-10)).sum(dim=-1)   # (C,) or (1,)
    return entropy.mean()


def spectral_entropy_map(
    image: torch.Tensor,
    patch_size: int = 16,
    stride: int = 8,
) -> torch.Tensor:
    """Compute a spatial map of local spectral entropy using sliding windows.

    Args:
        image: (C, H, W) tensor
        patch_size: local patch size for entropy computation
        stride: stride between patch centers

    Returns:
        entropy_map: (1, H, W) tensor bilinearly upsampled to input resolution
    """
    C, H, W = image.shape

    # Guard: if image is smaller than patch, fall back to global entropy
    if H < patch_size or W < patch_size:
        val = spectral_entropy(image).detach()
        return val.expand(1, H, W)

    # Extract patches via unfold: (C, n_h, n_w, patch_size, patch_size)
    x = image.unsqueeze(0)  # (1, C, H, W)
    patches = F.unfold(x, kernel_size=patch_size, stride=stride)  # (1, C*P*P, L)
    L = patches.shape[-1]
    n_h = (H - patch_size) // stride + 1
    n_w = (W - patch_size) // stride + 1

    # Reshape to (L*C, P, P) for batched DCT
    patches = patches.squeeze(0).T.reshape(L, C, patch_size, patch_size)
    patches_flat = patches.reshape(L * C, patch_size, patch_size)

    # 2D DCT on all patches in one pass
    X1 = _dct_1d(patches_flat)                              # (L*C, P, P) - DCT along cols
    X2 = _dct_1d(X1.transpose(-1, -2)).transpose(-1, -2)   # (L*C, P, P) - DCT along rows
    dct_coeffs = X2.reshape(L, C, patch_size, patch_size)

    # Per-patch spectral entropy
    power = dct_coeffs ** 2                          # (L, C, P, P)
    power_flat = power.reshape(L, C, -1)             # (L, C, P*P)
    p = power_flat / (power_flat.sum(dim=-1, keepdim=True) + 1e-10)
    entropy = -(p * torch.log(p + 1e-10)).sum(dim=-1).mean(dim=-1)  # (L,)

    # Reshape to grid and upsample
    entropy_grid = entropy.detach().reshape(1, 1, n_h, n_w)
    entropy_map = F.interpolate(entropy_grid, size=(H, W), mode='bilinear', align_corners=False)
    return entropy_map.squeeze(0)  # (1, H, W)


def spectral_l1_loss(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 loss between DCT coefficients of rendered and GT images.

    Args:
        rendered: (C, H, W) rendered image
        gt: (C, H, W) ground truth image

    Returns:
        Scalar loss tensor.
    """
    dct_rendered = dct_2d(rendered)
    dct_gt = dct_2d(gt)
    return torch.abs(dct_rendered - dct_gt).mean()


def spectral_entropy_loss(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Absolute difference between spectral entropies of rendered and GT.

    Args:
        rendered: (C, H, W) rendered image
        gt: (C, H, W) ground truth image

    Returns:
        Scalar loss tensor.
    """
    h_rendered = spectral_entropy(rendered)
    h_gt = spectral_entropy(gt)
    return torch.abs(h_rendered - h_gt)


def spectral_loss(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    lambda_se: float,
    lambda_sf: float,
) -> torch.Tensor:
    """Combined spectral loss: entropy difference + DCT coefficient L1.

    Args:
        rendered: (C, H, W) rendered image
        gt: (C, H, W) ground truth image
        lambda_se: weight for spectral entropy difference term
        lambda_sf: weight for spectral L1 (DCT coefficient) term

    Returns:
        Scalar loss tensor.
    """
    loss_se = spectral_entropy_loss(rendered, gt)
    loss_sf = spectral_l1_loss(rendered, gt)
    return lambda_se * loss_se + lambda_sf * loss_sf
