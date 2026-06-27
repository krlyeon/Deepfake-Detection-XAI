import streamlit as st
from PIL import Image

from src.gradcam import process_image

# Streamlit
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️",
    layout="wide"
)
st.title("🕵️ Deepfake Detection with XAI")
st.write(
    "딥페이크를 구분할 이미지를 업로드하세요."
)


# Model
@st.cache_resource
def load_model():
    model = get_model().to(device)
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    return model

# Upload
uploaded_file = st.file_uploader(
    "이미지 선택",
    type=["jpg", "jpeg", "png"]
)

# Prediction
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    pred, confidence, image, cam_image = process_image(image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            cam_image,
            caption="Grad-CAM",
            use_container_width=True
        )

    class_names = [
        "fake",
        "real"
    ]
    st.divider()
    st.subheader("Prediction")
    st.success(
        f"Prediction : {class_names[pred]}"
    )
    st.write(
        f"Confidence : {confidence*100:.2f}%"
    )