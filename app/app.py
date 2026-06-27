import streamlit as st

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ Deepfake Detection with XAI")

st.write(
    "딥페이크를 구분할 사진을 올려주세요."
)

uploaded_file = st.file_uploader(
    "이미지 선택",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Original Image",
        width=350
    )