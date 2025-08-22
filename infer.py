
import argparse, torch, os
from PIL import Image
from src.models.networks import ResnetGenerator
from src.utils.image import load_image, tensor_to_pil

def load_gen(weights_path: str, device: str = "cpu", size: int = 256):
    net = ResnetGenerator(3, 3, ngf=64, n_blocks=9)
    sd = torch.load(weights_path, map_location=device)
    # support state dict or full object
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = {k.replace("module.", ""): v for k, v in sd["state_dict"].items()}
    elif isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net.to(device).eval()
    return net

@torch.inference_mode()
def run(weights: str, input_path: str, output_path: str, size: int = 256, device: str = "cpu"):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    G = load_gen(weights, device, size)
    x, _ = load_image(input_path, size=size)
    x = x.to(device)
    y = G(x)
    out = tensor_to_pil(y)
    out.save(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to generator .pth (horse2zebra or zebra2horse)")
    ap.add_argument("--input", required=True, help="Input image")
    ap.add_argument("--output", required=True, help="Output image path")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.weights, args.input, args.output, args.size, args.device)
