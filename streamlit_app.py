# streamlit_app.py
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import VAE
from utils import postprocess_tensor

st.set_page_config(page_title="Image Restoration using VAE")

st.title("🛠️ Image Restoration with VAE")
st.write("Hello, Streamlit is working!")

uploaded_file = st.file_uploader("Upload a degraded image", type=["png", "jpg", "jpeg"])


# Load model
@st.cache_resource
def load_model():
    model = VAE(latent_dim=128)
    model.load_state_dict(torch.load("vae.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

@torch.no_grad()
def restore_image(img):
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 128, 128]
    restored_tensor = model(img_tensor)
    restored_image = postprocess_tensor(restored_tensor)
    return restored_image

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Degraded Input", use_column_width=True)

    try:
        with st.spinner("Restoring image..."):
            output_image = restore_image(input_image)
        st.image(output_image, caption="Restored Output", use_column_width=True)
        st.success("Done!")
    except Exception as e:
        st.error(f"Error during restoration: {e}")

