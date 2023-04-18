import torch.nn as nn
from torchvision import models

def create_model(model_name: str,
                fine_tune=None,
                num_classes=196,
                **kwargs):

    model = None

    # DenseNet
    if model_name == 'densenet_201': # sudah
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        model.name = 'densenet_201'
        model.classifier = nn.Linear(model.classifier.in_features, out_features=num_classes)
    
    # EfficientNet V1
    elif model_name == 'efficientnet_b1': # sudah
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        model.name = 'efficientnet_b1'
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    
    elif model_name == 'efficientnet_b4': # sudah
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        model.name = 'efficientnet_b4'
        model.classifier[1] = nn.Linear(in_features=1792, out_features=num_classes)

    # EfficientNet V2
    elif model_name == 'efficientnet_v2_s': # sudah
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        model.name = 'efficientnet_v2_s'
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    
    elif model_name == 'efficientnet_v2_m': # belum
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        model.name = 'efficientnet_v2_m'
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    
    elif model_name == 'efficientnet_v2_l': # belum
        model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        model.name = 'efficientnet_v2_l'
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    # ResNet
    elif model_name == 'resnet_50': # sudah
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.name = 'resnet_50'
        model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
    
    elif model_name == 'resnet_34': # sudah
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.name = 'resnet_34'
        model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
    
    # MobileNet
    elif model_name == 'mobilenet_v3_l': # sudah
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.name = 'mobilenet_v3_l'
        model.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes)
        
    # ShuffleNet
    elif model_name == 'shufflenet_v2_x2': # sudah
        model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT)
        model.name = 'shufflenet_v2_x2'
        model.fc = nn.Linear(model.fc.in_features , out_features=num_classes)
        
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