import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models


nclasses = 250


class Net(nn.Module):
    def __init__(self,num_classes=nclasses):
        super(Net, self).__init__()
        resnet = models.resnet18(pretrained=True)
        
        # Remove the fully connected layer of ResNet-18
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add your custom fully connected layer
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
