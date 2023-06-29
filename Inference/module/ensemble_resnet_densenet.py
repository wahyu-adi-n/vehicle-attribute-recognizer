import torch
import torch.nn as nn

class EnsembleResDenseNet(nn.Module):   
    def __init__(self, res_net, 
                        dense_net, 
                        input_size: int,
                        num_classes: int= 196):
        super().__init__()
        self.modelA = res_net
        self.modelB = dense_net
        self.classifier = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x = torch.cat((x1, x2), dim=1)
        out = self.classifier(x)
        return out