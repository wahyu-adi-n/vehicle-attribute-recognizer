import torch.nn as nn
from torchvision import models


def create_model(model_name: str,
                fine_tune=True,
                num_classes=196,
                 **kwargs):

    model = None

    if model_name == 'densenet201':
        model = models.densenet201(weights='IMAGENET1K_V1')

        model.classifier = nn.Linear(
          in_features=1920, out_features=num_classes
        )

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes
        )

    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights='IMAGENET1K_V2')
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes
        )

    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes
        )

    elif model_name == 'inceptionv3':
        model = models.inception_v3(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(
            in_features=2048, out_features=num_classes
        )

    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2')
        model.fc = nn.Linear(
            in_features=2048, out_features=num_classes
        )

    elif model_name == 'shufflenetv2x2':
        model = models.shufflenet_v2_x2_0(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(in_features=2048, 
                            out_features=num_classes)

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
