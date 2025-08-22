
from __future__ import annotations
import torch, io
from PIL import Image
import numpy as np
from typing import Tuple

def load_image(path_or_bytes, size: int = 256) -> Tuple[torch.Tensor, Image.Image]:
    """Load image as normalized tensor in [-1,1], CHW."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(path_or_bytes)).convert("RGB")
    else:
        img = Image.open(path_or_bytes).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return ten, img

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[-1,1] CHW -> PIL"""
    t = t.detach().cpu().clamp(-1, 1)
    t = (t * 0.5 + 0.5)  # [0,1]
    t = (t * 255.0).round().byte()
    if t.dim() == 4:
        t = t[0]
    arr = t.permute(1, 2, 0).numpy()
    return Image.fromarray(arr)

def make_grid(left: Image.Image, right: Image.Image) -> Image.Image:
    w = left.width + right.width
    h = max(left.height, right.height)
    out = Image.new("RGB", (w, h))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    return out
