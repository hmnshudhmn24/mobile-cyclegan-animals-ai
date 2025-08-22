
# 🦓 Mobile CycleGAN Animal-to-Animal Style Transfer (Horse ⇄ Zebra)

Turn **horses → zebras** (and back again) on your **phone** or desktop!  
This repo bundles a **mobile-friendly Streamlit UI**, a clean **PyTorch CycleGAN generator**, tiny **demo trainer**, and **TorchScript export** for mobile runtimes.

---

## ✨ Highlights

- 📱 **Mobile-friendly UI**: Works with your phone’s camera via `st.camera_input`.
- 🔁 **CycleGAN-style**: ResNet-9 generator + PatchGAN discriminator.
- ⚡ **Fast inference**: 256–384px images run comfortably on CPU.
- 🧠 **DIY training (lite)**: A minimal trainer for toy datasets.
- 📦 **Mobile export**: Export the generator to **TorchScript** for embedding.

---

## 🚀 Quickstart

```bash
# 1) Install
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Get weights (place files in weights/)
#   - horse2zebra_gen.pth
#   - zebra2horse_gen.pth
#   (Use your own checkpoints, train with train_lite.py, or convert from existing sources.)

# 3) Run mobile UI
streamlit run app.py
```

Open the URL shown in the terminal. On mobile, open the same URL and use the **Camera** tab 📷.

> ✅ If you don’t have weights yet, run a tiny demo training:
> ```bash
> python train_lite.py --data data --epochs 5 --size 256 --bs 2
> ```
> This expects `data/trainA/` (e.g., horses) and `data/trainB/` (e.g., zebras).

---

## 🧩 Folder Layout

```
mobile-cyclegan-animals-ai/
├─ app.py                 # Streamlit app (camera + upload)
├─ infer.py               # CLI inference
├─ export_torchscript.py  # TorchScript export for mobile embedding
├─ train_lite.py          # Tiny CycleGAN trainer (educational)
├─ src/models/networks.py # ResNet-9 G + PatchGAN D
├─ src/utils/image.py     # I/O, transforms, utilities
├─ weights/               # Put your .pth here
└─ requirements.txt
```

---

## 🖥️ Streamlit App (Mobile-first)

- **Camera** tab for live capture on phones.
- **Upload** tab for images.
- **Direction switch**: Horse → Zebra or Zebra → Horse.
- **Image size slider**: 128–384 px (balance speed/quality).
- **Download** button for the result.

```bash
streamlit run app.py
```

---

## ⚙️ Inference via CLI

```bash
python infer.py   --weights weights/horse2zebra_gen.pth   --input examples/horse.jpg   --output out/zebra.png   --size 256 --device cpu
```

---

## 🧪 Tiny Trainer (Educational)

> ⚠️ This is a **lite** training loop meant for learning & demos.  
> For best results, use a mature CycleGAN training script (identity loss, image buffer, schedulers, flipping, etc.).

### Data layout
```
data/
  trainA/   # domain A (e.g., horse)
  trainB/   # domain B (e.g., zebra)
```

### Run
```bash
python train_lite.py --data data --epochs 5 --size 256 --bs 2
# Saves:
#   weights/horse2zebra_gen.pth
#   weights/zebra2horse_gen.pth
```

**Tips**
- Increase `--epochs` (e.g., 100+) and use a **GPU** for better quality.
- Add identity loss and learning-rate schedulers if you extend the trainer.

---

## 📲 Mobile Export (TorchScript)

Embed the generator in mobile apps:

```bash
python export_torchscript.py   --weights weights/horse2zebra_gen.pth   --out weights/horse2zebra_gen.ts   --size 256
```

Load `*.ts` with **PyTorch Mobile** or a native wrapper.

---

## 🧠 Model Notes

- **Generator**: ResNet-9 blocks (Reflection padding, InstanceNorm, Tanh output).
- **Discriminator**: 70×70 PatchGAN (BCEWithLogits).
- **Preprocess**: Resize to square (128–384), normalize to `[-1,1]`.
- **Postprocess**: Denormalize to `[0,255]`, PNG output.

---

## 🛡️ Ethics & Safety

- This is a **visual style translation** demo, not identity morphing.
- Clearly label outputs as **synthetic** in downstream apps.
- Avoid misuse in contexts where realism could mislead users.

---

## 🔧 Troubleshooting

- **Missing weights**: Put `horse2zebra_gen.pth` / `zebra2horse_gen.pth` in `weights/`.
- **CUDA OOM**: Lower `--size` to 256 (or 128).
- **Slow on CPU**: Try 128–224 px; use TorchScript for speedups.

---

## 🗺️ Roadmap

- [ ] Identity loss & image replay buffer in trainer  
- [ ] Mixed-precision inference  
- [ ] WebGPU/WebAssembly demo  
- [ ] On-device iOS/Android inference sample app

---

## 📜 License

MIT — have fun, share your animal mashups, and drop a ⭐ if you like it!
