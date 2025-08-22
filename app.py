
import os, io, torch
import streamlit as st
from PIL import Image
from src.models.networks import ResnetGenerator
from src.utils.image import load_image, tensor_to_pil, make_grid

st.set_page_config(page_title="Mobile CycleGAN – Animal ↔ Animal", page_icon="🦓", layout="centered")

@st.cache_resource
def load_gen(weights_path: str, device: str = "cpu"):
    net = ResnetGenerator(3,3,ngf=64,n_blocks=9)
    sd = torch.load(weights_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("module.",""): v for k,v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net.eval().to(device)
    return net

def ensure_weights():
    os.makedirs("weights", exist_ok=True)
    needed = ["horse2zebra_gen.pth", "zebra2horse_gen.pth"]
    missing = [n for n in needed if not os.path.exists(os.path.join("weights", n))]
    return missing

st.title("🦓 Mobile CycleGAN: Animal ↔ Animal (Horse ⇄ Zebra)")
st.caption("Use your phone camera 📷 or upload an image. The app runs a CycleGAN generator to swap animal style.")

device = "cpu"  # server-friendly default
size = st.sidebar.select_slider("Image size (px)", options=[128,192,224,256,320,384], value=256)
direction = st.sidebar.radio("Direction", ["Horse → Zebra (G_AB)","Zebra → Horse (G_BA)"], index=0)
weights_name = "horse2zebra_gen.pth" if "Horse" in direction and "Zebra" in direction else "zebra2horse_gen.pth"
weights_path = os.path.join("weights", weights_name)

# Info box about weights
missing = ensure_weights()
if weights_name in missing:
    st.warning(f"Missing weights: `{weights_name}` in `weights/`.\n\n➡️ Place a pretrained generator at `weights/{weights_name}`.\nYou can also train a tiny demo with `python train_lite.py`.")
else:
    st.success(f"Loaded weights from `weights/{weights_name}` (or ready to load).")

# Input: camera or upload
tab1, tab2 = st.tabs(["📷 Camera", "🖼️ Upload"])
img_bytes = None
with tab1:
    cam = st.camera_input("Take a photo (works on mobile)")
    if cam is not None:
        img_bytes = cam.getvalue()
with tab2:
    up = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if up is not None:
        img_bytes = up.read()

go = st.button("✨ Transform")

if go and img_bytes is not None and not (weights_name in missing):
    with st.spinner("Running generator..."):
        G = load_gen(weights_path, device=device)
        x, pil_in = load_image(io.BytesIO(img_bytes), size=size)
        with torch.inference_mode():
            y = G(x.to(device))
        pil_out = tensor_to_pil(y)
        grid = make_grid(pil_in.resize((size,size)), pil_out.resize((size,size)))
        st.image(grid, caption="Left: Input • Right: Output", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        pil_out.save(buf, format="PNG")
        st.download_button("⬇️ Download output", buf.getvalue(), file_name="cyclegan_output.png", mime="image/png")
elif go and img_bytes is None:
    st.info("Please capture a photo or upload an image first.")
