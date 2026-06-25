from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import random

# ======================================================
# 경로 설정
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "mini"

random.seed(42)

# ======================================================
# 데이터 개수 확인
# ======================================================
def count_images():

    print("=" * 60)
    print("데이터 개수")
    print("=" * 60)

    for split in ["train", "valid", "test"]:
        real = len(list((DATA_DIR / split / "real").glob("*")))
        fake = len(list((DATA_DIR / split / "fake").glob("*")))
        print(
            f"{split:<5} | Real : {real:<5} | Fake : {fake:<5} | Total : {real+fake}"
        )


# ======================================================
# 이미지 크기 확인
# ======================================================
def check_image_size():
    print("\n" + "=" * 60)
    print("이미지 크기 확인")
    print("=" * 60)

    sizes = []
    for split in ["train", "valid", "test"]:
        for label in ["real", "fake"]:
            images = list((DATA_DIR / split / label).glob("*"))

            for image_path in random.sample(images, min(30, len(images))):
                img = Image.open(image_path)
                sizes.append(img.size)

    counter = Counter(sizes)

    print("\n가장 많이 등장한 이미지 크기")

    for size, count in counter.most_common(10):
        print(f"{size} : {count}장")

# ======================================================
# 샘플 이미지 출력
# ======================================================

def show_samples():
    print("\n샘플 이미지 출력")
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    images = []

    for label in ["real", "fake"]:
        image_list = list((DATA_DIR / "train" / label).glob("*"))
        images.extend(random.sample(image_list, 8))
    random.shuffle(images)

    for ax, image_path in zip(axes.flatten(), images):
        img = Image.open(image_path)
        ax.imshow(img)
        ax.set_title(image_path.parent.name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ======================================================
# 메인
# ======================================================
def main():

    count_images()
    check_image_size()
    show_samples()

if __name__ == "__main__":
    main()