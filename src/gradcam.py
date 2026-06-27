from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import test_dataset, test_transform
from model import get_model
import matplotlib.pyplot as plt


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "outputs" / "best_model.pth"
OUTPUT_DIR = BASE_DIR / "outputs" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"\n사용 장치 : {device}")


# 모델 불러오기
model = get_model().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# Grad-CAM target layer
target_layers = [
    model.features[-2]
]


# 이미지 1장 선택
image_path, true_label = test_dataset.samples[0]
image = Image.open(image_path).convert("RGB")
input_tensor = test_transform(image).unsqueeze(0).to(device)


# 예측
output = model(input_tensor)
prob = torch.softmax(output, dim=1)
pred_label = output.argmax(dim=1).item()
confidence = prob[0][pred_label].item()


# Grad-CAM 생성
targets = [
    ClassifierOutputTarget(pred_label)
]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]


# 원본 이미지를 0~1 범위 numpy로 변환
rgb_image = np.array(
    image.resize((224, 224))
).astype(np.float32) / 255.0


cam_image = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)

# Original + Grad-CAM 비교 이미지 저장

def save_result_image(
    original_image,
    gradcam_image,
    true_name,
    pred_name,
    confidence,
    save_path
):

    result = "✓ Correct" if true_name == pred_name else "✗ Wrong"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )

    # Original
    axes[0].imshow(original_image)
    axes[0].set_title("Original", fontsize=15)
    axes[0].axis("off")

    # Grad-CAM
    axes[1].imshow(gradcam_image)
    axes[1].set_title("Grad-CAM", fontsize=15)
    axes[1].axis("off")

    # 결과 정보
    info = (
        f"Ground Truth : {true_name}\n"
        f"Prediction   : {pred_name}\n"
        f"Confidence   : {confidence*100:.2f}%\n"
        f"Result       : {result}"
    )

    plt.figtext(
        0.5,
        0.01,
        info,
        ha="center",
        va="bottom",
        fontsize=11,
        family="monospace"
    )

    plt.tight_layout(rect=[0, 0.15, 1, 1])

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
# 저장
save_path = OUTPUT_DIR / "sample_gradcam.png"

save_result_image(
    original_image=image.resize((224, 224)),
    gradcam_image=cam_image,
    true_name=test_dataset.classes[true_label],
    pred_name=test_dataset.classes[pred_label],
    confidence=confidence,
    save_path=save_path
)

print(f"True Label      : {test_dataset.classes[true_label]}")
print(f"Predicted Label : {test_dataset.classes[pred_label]}")
print(f"Confidence      : {confidence:.4f}")
print(f"Grad-CAM 저장   : {save_path}")