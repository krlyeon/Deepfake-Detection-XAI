from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights

import torch.nn as nn


# 모델 생성
def get_model(num_classes=2):
    """
    EfficientNet-B0 모델 생성
    """
    # ImageNet으로 사전학습된 가중치 사용
    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    # 마지막 분류층만 현재 프로젝트에 맞게 변경 (Real / Fake)
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


# 테스트
if __name__ == "__main__":
    model = get_model()
    print(model)