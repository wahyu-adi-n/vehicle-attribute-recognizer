import torch.nn as nn
from torchvision import models

def create_model(model_name: str,
                fine_tune: bool =False,
                num_classes: int = 200):

    model = None

    if model_name == 'densenet_201':
        weights = models.DenseNet201_Weights.DEFAULT
        model = models.densenet201(weights=weights)
        model.name = 'densenet_201'
        model.classifier = nn.Linear(in_features=model.classifier.in_features, 
                      out_features=num_classes)
    
    elif model_name == 'efficientnet_b4':
        weights = models.EfficientNet_B4_Weights.DEFAULT
        model = models.efficientnet_b4(weights=weights)
        model.name = 'efficientnet_b4'
        model.classifier[1] = nn.Linear(in_features=1792, 
                                        out_features=num_classes)

    elif model_name == 'efficientnet_v2_s':
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)
        model.name = 'efficientnet_v2_s'
        model.classifier[1] = nn.Linear(in_features=1280, 
                                        out_features=num_classes)

    elif model_name == 'resnet_50':
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.name = 'resnet_50'
        
        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False
            
            model.fc = nn.Linear(in_features=model.fc.in_features, 
                            out_features=num_classes)
                
            return model

        model.fc = nn.Linear(in_features=model.fc.in_features, 
                            out_features=num_classes)      

    elif model_name == 'mobilenet_v3_l':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        model.name = 'mobilenet_v3_l'
        model.classifier[3] = nn.Linear(in_features=1280, 
                                        out_features=num_classes)
        
    elif model_name == 'shufflenet_v2_x2':
        weights = models.ShuffleNet_V2_X2_0_Weights.DEFAULT
        model = models.shufflenet_v2_x2_0(weights=weights)
        model.name = 'shufflenet_v2_x2'
        model.fc = nn.Linear(in_features=model.fc.in_features , 
                            out_features=num_classes)
    
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

    return model