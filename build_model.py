import torch
import torch.nn as nn
from torchvision import models

def build_mobilenet_v2(num_classes, fine_tune=False, use_pretrained=True):
    # Tải model với hỗ trợ cả API cũ và mới của torchvision
    try:
        # torchvision mới: weights enum
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.mobilenet_v2(weights=weights)
    except Exception:
        # fallback cho torchvision cũ
        model = models.mobilenet_v2(pretrained=use_pretrained)

    # Nếu không fine-tune, chỉ train classifier
    if not fine_tune:
        for param in model.features.parameters():
            param.requires_grad = False

    # Lấy số chiều input của classifier một cách an toàn
    if hasattr(model, "last_channel"):
        in_features = model.last_channel
    else:
        try:
            in_features = model.classifier[1].in_features
        except Exception:
            in_features = 1280  # giá trị mặc định của MobileNetV2

    # Thay classifier cuối cùng bằng tầng mới và khởi tạo trọng số
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(in_features, num_classes)
    )
    # Khởi tạo weights cho lớp Linear mới
    nn.init.normal_(model.classifier[1].weight, 0, 0.01)
    nn.init.zeros_(model.classifier[1].bias)

    return model
