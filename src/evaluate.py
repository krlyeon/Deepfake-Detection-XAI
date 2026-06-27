from pathlib import Path
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from dataset import test_loader
from model import get_model


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "best_model.pth"


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


# Evaluation
def evaluate():

    y_true = []
    y_pred = []

    correct_samples = []
    wrong_samples = []

    with torch.no_grad():
        image_idx = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(
                labels.cpu().numpy()
            )
            y_pred.extend(
                preds.cpu().numpy()
            )

            # Grad-CAM에서 사용할 정답/오답 샘플 저장
            for i in range(len(labels)):
                if preds[i] == labels[i]:
                    correct_samples.append(image_idx)
                else:
                    wrong_samples.append(image_idx)
                image_idx += 1

    return (
        y_true,
        y_pred,
        correct_samples,
        wrong_samples
    )


# 결과 출력
def print_result(
    y_true,
    y_pred,
    correct_samples,
    wrong_samples
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    print("=" * 50)
    print("Test Result")
    print("=" * 50)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print()
    print("=" * 50)
    print("Confusion Matrix")
    print("=" * 50)

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    print()
    print("=" * 50)
    print("Classification Report")
    print("=" * 50)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["fake", "real"]
        )
    )

    print()

    print("=" * 50)
    print("Grad-CAM Sample")
    print("=" * 50)
    print(f"Correct Sample : {len(correct_samples)}")
    print(f"Wrong Sample   : {len(wrong_samples)}")


# Main
def main():
    y_true, y_pred, correct_samples, wrong_samples = evaluate()
    print_result(
        y_true,
        y_pred,
        correct_samples,
        wrong_samples
    )


if __name__ == "__main__":
    main()