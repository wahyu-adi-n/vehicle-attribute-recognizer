import torch.nn as nn
from torchvision import models

def create_model(model_name: str,
                fine_tune: bool,
                num_classes: int = 196):

    model = None

    # DenseNet
    if model_name == 'densenet_201': # sudah
        weights = models.DenseNet201_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.densenet201(weights=weights)
        model.name = 'densenet_201'
        model.classifier = nn.Linear(in_features=model.classifier.in_features, 
                                    out_features=num_classes)
    
    # EfficientNet V1
    elif model_name == 'efficientnet_b1': # sudah
        weights = models.EfficientNet_B1_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.efficientnet_b1(weights=weights)
        model.name = 'efficientnet_b1'
        model.classifier[1] = nn.Linear(in_features=1280, 
                                        out_features=num_classes)
    
    elif model_name == 'efficientnet_b4': # sudah
        weights = models.EfficientNet_B4_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.efficientnet_b4(weights=weights)
        model.name = 'efficientnet_b4'
        model.classifier[1] = nn.Linear(in_features=1792, 
                                        out_features=num_classes)

    # EfficientNet V2
    elif model_name == 'efficientnet_v2_s': # sudah
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.efficientnet_v2_s(weights=weights)
        model.name = 'efficientnet_v2_s'
        model.classifier[1] = nn.Linear(in_features=1280, 
                                        out_features=num_classes)

    # ResNet
    elif model_name == 'resnet_50': # sudah
        weights = models.ResNet50_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.resnet50(weights=weights)
        model.name = 'resnet_50'
        model.fc = nn.Linear(in_features=model.fc.in_features, 
                            out_features=num_classes)
    
    elif model_name == 'resnet_34': # sudah
        weights = models.ResNet34_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.resnet34(weights=weights)
        model.name = 'resnet_34'
        model.fc = nn.Linear(in_features=model.fc.in_features, 
                            out_features=num_classes)
    
    # MobileNet
    elif model_name == 'mobilenet_v3_l': # sudah
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.mobilenet_v3_large(weights=weights)
        model.name = 'mobilenet_v3_l'
        model.classifier[3] = nn.Linear(in_features=1280, 
                                        out_features=num_classes)
        
    # ShuffleNet
    elif model_name == 'shufflenet_v2_x2': # sudah
        weights = models.ShuffleNet_V2_X2_0_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.shufflenet_v2_x2_0(weights=weights)
        model.name = 'shufflenet_v2_x2'
        model.fc = nn.Linear(in_features=model.fc.in_features , 
                            out_features=num_classes)
    
    elif model_name == 'vit_b_16': # sudah
        weights = models.ViT_B_16_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.vit_b_16(weights=weights)
        model.name = 'vit_b_16'
        model.heads = nn.Sequential(
                          nn.Linear(in_features=768, 
                            out_features=num_classes)
                      )
    else:
        raise NotImplementedError

    if fine_tune:
        print('[INFO]: Unfreezing all layers...')
        for params in model.parameters():
            params.requires_grad = True

    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.features.parameters():
            params.requires_grad = False

    return model, model_transform