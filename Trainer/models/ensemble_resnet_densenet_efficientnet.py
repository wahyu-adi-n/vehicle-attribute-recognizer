import torch
import torch.nn as nn

class EnsembleResDenseEfficientNet(nn.Module):   
    def __init__(self, res_net,
                        dense_net, 
                        efficient_net, 
                        input_size: int,
                        num_classes: int= 196):
        super().__init__()
        self.modelA = res_net
        self.modelB = dense_net
        self.modelC = efficient_net
        self.classifier = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.classifier(x)
        return out