from pathlib import Path

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# CUDA에서 데이터 전송 속도를 높이는 옵션
import torch
PIN_MEMORY = torch.cuda.is_available()


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "mini"

# 하이퍼파라미터
IMAGE_SIZE = 224
BATCH_SIZE = 32
# # 데이터를 불러오는 프로세스 개수
NUM_WORKERS = 4


# 이미지 전처리
# (데이터 증강 포함)
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset
train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    DATA_DIR / "valid",
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=test_transform
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

# 클래스 확인
def print_dataset_info():

    print("=" * 60)
    print("Dataset 정보")
    print("=" * 60)

    print(f"Class : {train_dataset.classes}")
    print(f"Train : {len(train_dataset)}")
    print(f"Valid : {len(valid_dataset)}")
    print(f"Test  : {len(test_dataset)}")

    print()

    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape : {images.shape}")
    print(f"Batch Label Shape : {labels.shape}")


if __name__ == "__main__":
    print_dataset_info()