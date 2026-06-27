from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import test_dataset, test_transform
from src.model import get_model
import random

random.seed(42)

# Path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "best_model.pth"
OUTPUT_DIR = BASE_DIR / "outputs" / "gradcam"


# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"\n사용 장치 : {device}")


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


# Grad-CAM
cam = GradCAM(
    model=model,
    target_layers=[model.features[-2]]
)

# Result Image
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
        figsize=(11, 6)
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gradcam_image)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    info = (
        f"Ground Truth : {true_name}\n"
        f"Prediction   : {pred_name}\n"
        f"Confidence   : {confidence*100:.2f}%\n"
        f"Result       : {result}"
    )

    plt.figtext(
        0.5,
        0.02,
        info,
        ha="center",
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


# Process One Image
def process_image(image):

    image = image.convert("RGB")
    image = image.resize((224,224))

    input_tensor = test_transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    prob = torch.softmax(output, dim=1)
    pred_label = output.argmax(dim=1).item()
    confidence = prob[0][pred_label].item()

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[
            ClassifierOutputTarget(pred_label)
        ]
    )[0]

    rgb_image = np.array(image).astype(np.float32) / 255.0

    cam_image = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return pred_label, confidence, image, cam_image

# Save Directory
FAKE_CORRECT_DIR = OUTPUT_DIR / "fake_correct"
FAKE_WRONG_DIR = OUTPUT_DIR / "fake_wrong"
REAL_CORRECT_DIR = OUTPUT_DIR / "real_correct"
REAL_WRONG_DIR = OUTPUT_DIR / "real_wrong"

for folder in [
    FAKE_CORRECT_DIR,
    FAKE_WRONG_DIR,
    REAL_CORRECT_DIR,
    REAL_WRONG_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


def get_save_dir(true_name, pred_name):

    if true_name == "fake":
        if pred_name == "fake":
            return FAKE_CORRECT_DIR
        return FAKE_WRONG_DIR
    else:
        if pred_name == "real":
            return REAL_CORRECT_DIR
        return REAL_WRONG_DIR


# 배치용 함수
def process_image_from_path(image_path):
    image = Image.open(image_path)
    return process_image(image)


# Main
def main():
    samples = random.sample(
    test_dataset.samples,
    500
    )
    for idx, (image_path, true_label) in enumerate(samples):

        true_name = test_dataset.classes[true_label]
        pred_label, confidence, image, cam_image = process_image_from_path(image_path)
        pred_name = test_dataset.classes[pred_label]

        save_dir = get_save_dir(
            true_name,
            pred_name
        )

        save_path = save_dir / (
            f"{idx:03d}_{true_name}_pred_{pred_name}.png"
        )

        save_result_image(
            original_image=image,
            gradcam_image=cam_image,
            true_name=true_name,
            pred_name=pred_name,
            confidence=confidence,
            save_path=save_path
        )

        print(
            f"[{idx+1}/{len(samples)}] {save_path.name}"
        )

    print("\nGrad-CAM 생성 완료!")


if __name__ == "__main__":
    main()