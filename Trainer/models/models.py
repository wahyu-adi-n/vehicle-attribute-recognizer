import torch.nn as nn
from torchvision import models


def create_model(model_name: str,
                fine_tune=None,
                num_classes=196,
                 **kwargs):

    model = None

    # DenseNet
    if model_name == 'densenet_201': # train lagi, weight decay 0.001
        model = models.densenet201(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(in_features=1920, out_features=num_classes)
    
    # EfficientNet V1
    elif model_name == 'efficientnet_b1': # sudah
        model = models.efficientnet_b1(weights='IMAGENET1K_V2')
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes
        )
    
    elif model_name == 'efficientnet_b4': # belum
        model = models.efficientnet_b4(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes
        )

    # EfficientNet V2
    elif model_name == 'efficientnet_v2_s': # sudah
        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes
        )

    # ResNet
    elif model_name == 'resnet_50': # sudah
        model = models.resnet50(weights='IMAGENET1K_V2')
        model.fc = nn.Linear(
            in_features=2048, out_features=num_classes
        )
    
    # MobileNet
    elif model_name == 'mobilenet_v3_l': # belum
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V2')
        model.classifier[3] = nn.Linear(
            in_features=1280, out_features=num_classes
        )
        
    # ShuffleNet
    elif model_name == 'shufflenet_v2_x2': # belum
        model = models.shufflenet_v2_x2_0(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(
            in_features=2048, out_features=num_classes
        )
        
    else:
        raise NotImplementedError

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True

    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    return model
