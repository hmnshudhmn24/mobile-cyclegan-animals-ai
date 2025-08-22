
import argparse, torch
from src.models.networks import ResnetGenerator

def export(weights: str, out_path: str, size: int = 256, device: str = "cpu"):
    G = ResnetGenerator(3,3,ngf=64,n_blocks=9)
    sd = torch.load(weights, map_location="cpu")
    if "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("module.",""): v for k,v in sd.items()}
    G.load_state_dict(sd, strict=False)
    G.eval().to(device)
    example = torch.randn(1,3,size,size, device=device)
    ts = torch.jit.trace(G, example)
    ts.save(out_path)
    print(f"Saved TorchScript: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="generator.ts")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    export(args.weights, args.out, args.size, args.device)
