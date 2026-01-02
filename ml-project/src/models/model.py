import torch.nn as nn
from torchvision import models


def build_efficientnet(num_classes):
    """
    Build EfficientNet-B0 with frozen backbone.
    """

    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
