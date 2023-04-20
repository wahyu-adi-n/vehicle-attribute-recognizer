import torch
import torch.nn as nn

class EnsembleModel(nn.Module):   
    def __init__(self, modelA, modelB, modelC, num_classes:int=196):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.classifier = nn.Linear(num_classes * 3, num_classes)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.classifier(x)
        return out