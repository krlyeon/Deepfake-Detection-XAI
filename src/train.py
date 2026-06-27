from pathlib import Path
import random

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, valid_loader
from model import get_model


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "best_model.pth"
HISTORY_PATH = OUTPUT_DIR / "history.csv"
GRAPH_PATH = OUTPUT_DIR / "history.png"


# 하이퍼파라미터
EPOCHS = 10
LEARNING_RATE = 1e-4
PATIENCE = 3
SEED = 42


# Seed 고정
random.seed(SEED)
torch.manual_seed(SEED)


# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"\n사용 장치 : {device}")



# 모델
model = get_model().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# History
train_loss_history = []
valid_loss_history = []
valid_acc_history = []


# Validation
def evaluate():

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(valid_loader), correct / total


# Train
def train():

    best_acc = 0
    patience = 0

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        train_acc = correct / total

        valid_loss, valid_acc = evaluate()
        scheduler.step(valid_loss)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nTrain Loss : {train_loss:.4f}"
            f" | Train Acc : {train_acc:.4f}"
            f" | Valid Loss : {valid_loss:.4f}"
            f" | Valid Acc : {valid_acc:.4f}"
            f" | LR : {current_lr:.6f}"
        )

        if valid_acc > best_acc:
            best_acc = valid_acc
            patience = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": best_acc
                },
                MODEL_PATH
            )
            print("Best Model 저장")

        else:
            patience += 1

        if patience >= PATIENCE:
            print("\nEarly Stopping")
            break


# 결과 저장
def save_history():
    df = pd.DataFrame({
        "epoch": range(1, len(train_loss_history) + 1),
        "train_loss": train_loss_history,
        "valid_loss": valid_loss_history,
        "valid_accuracy": valid_acc_history
    })

    df.to_csv(
        HISTORY_PATH,
        index=False
    )

    epochs = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, label="Train")
    plt.plot(epochs, valid_loss_history, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, valid_acc_history)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")

    plt.tight_layout()
    plt.savefig(GRAPH_PATH)
    plt.close()


# Main
def main():
    train()
    save_history()
    print("\n학습 완료")
    print(f"Best Model : {MODEL_PATH}")
    print(f"History CSV : {HISTORY_PATH}")
    print(f"History Graph : {GRAPH_PATH}")


if __name__ == "__main__":
    main()