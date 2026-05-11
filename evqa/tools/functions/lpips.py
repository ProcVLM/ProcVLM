import torch
import lpips
import numpy as np
from PIL import Image
from typing import Optional

_cached_lpips_model = None
_cached_device = None


def get_lpips_model(device: Optional[str] = None):
    """
    Build and cache LPIPS model.
    """
    global _cached_lpips_model, _cached_device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _cached_lpips_model is None or _cached_device != device:
        model = lpips.LPIPS(net='alex') # use alex backbone for minimal computation
        model = model.to(device)
        model.eval()

        _cached_lpips_model = model
        _cached_device = device

    return _cached_lpips_model


def _preprocess(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL.Image to normalized torch tensor for LPIPS.
    Output shape: (1, 3, H, W), range [-1, 1]
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = np.array(img).astype(np.float32) / 255.0
    img = img * 2 - 1  # [0,1] -> [-1,1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def image_lpips_sim(
    img1: Image.Image,
    img2: Image.Image,
    device: Optional[str] = None,
    eps: float = 1e-6
) -> float:
    """
    Compute LPIPS-based similarity between two images.
    Returns:
        similarity in [0, 1], where 1 means very similar, 0 means very different.
    """
    model = get_lpips_model(device)
    device = _cached_device

    t1 = _preprocess(img1).to(device)
    t2 = _preprocess(img2).to(device)

    with torch.no_grad():
        dist = model(t1, t2).item()  # LPIPS distance, usually in [0, ~1]

    # convert distance to similarity
    # empirically, LPIPS rarely exceeds 1.0, so this mapping is stable
    sim = 1.0 / (1.0 + dist + eps)

    return float(sim)
