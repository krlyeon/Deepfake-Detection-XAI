import streamlit as st
from pathlib import Path

import torch

from PIL import Image
from src.model import get_model
from src.dataset import test_transform
from src.gradcam import process_image

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ Deepfake Detection with XAI")

st.write(
    "딥페이크를 구분할 사진을 올려주세요."
)

# Path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "best_model.pth"

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Model
model = get_model().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


uploaded_file = st.file_uploader(
    "이미지 선택",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            image,
            caption="Original Image",
            width=350
        )

    input_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)

    pred = output.argmax(dim=1).item()
    confidence = prob[0][pred].item()
    class_names = ["fake", "real"]

    st.divider()
    st.subheader("Prediction")
    st.success(f"Prediction : {class_names[pred]}")
    st.write(f"Confidence : {confidence * 100:.2f}%")