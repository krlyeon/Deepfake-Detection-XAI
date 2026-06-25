from pathlib import Path
import random
import os

# ======================================================
# 경로 설정
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_DIR = BASE_DIR / "data" / "original"
MINI_DIR = BASE_DIR / "data" / "mini"

# ======================================================
# 샘플 개수
# ======================================================
SAMPLE_SIZE = {
    "train": 2500,
    "valid": 500,
    "test": 500
}
# 랜덤 시드 고정 (재현 가능)
SEED = 42
random.seed(SEED)


# ======================================================
# mini 폴더 생성
# ======================================================
def create_folders():
    """mini 데이터셋 폴더 생성"""
    for split in SAMPLE_SIZE.keys():
        for label in ["real", "fake"]:
            (MINI_DIR / split / label).mkdir(
                parents=True,
                exist_ok=True
            )


# ======================================================
# 이미지 샘플링
# ======================================================
def sample_images(split: str, label: str):
    """
    원본 데이터에서 랜덤 샘플링 후
    심볼릭 링크를 생성
    """
    source_dir = ORIGINAL_DIR / split / label
    target_dir = MINI_DIR / split / label

    images = list(source_dir.glob("*"))

    if len(images) == 0:
        print(f"[경고] {source_dir} 에 이미지가 없습니다.")
        return

    sample_num = min(SAMPLE_SIZE[split], len(images))
    sampled_images = random.sample(images, sample_num)

    count = 0

    for image in sampled_images:
        target_path = target_dir / image.name
        # 이미 존재하면 건너뜀
        if target_path.exists():
            continue
        os.symlink(
            image.resolve(),
            target_path
        )
        count += 1

    print(
        f"{split:<5} | {label:<4} | "
        f"원본 {len(images):>6}장 → "
        f"샘플 {count:>4}장"
    )


# ======================================================
# 메인 함수
# ======================================================
def main():
    print("=" * 60)
    print("Mini Dataset 생성")
    print("=" * 60)

    create_folders()

    for split in SAMPLE_SIZE.keys():
        for label in ["real", "fake"]:
            sample_images(split, label)
    print("=" * 60)
    print("완료")
    print("=" * 60)

# ======================================================
# 실행
# ======================================================
if __name__ == "__main__":
    main()