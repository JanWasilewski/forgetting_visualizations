import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet18, self).__init__()
        original_model = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.fc = nn.Linear(original_model.fc.in_features, num_classes)
        
    def forward(self, x):
        penultimate_features = self.features(x)
        penultimate_features = torch.flatten(penultimate_features, 1)
        class_outputs = self.fc(penultimate_features)
        return penultimate_features, class_outputs

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.LazyConv2d(16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.LazyLinear(2)
        #self.bottleneck = nn.Linear(128, 2)  # Bottleneck layer
        self.fc2 = nn.Linear(2, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  

        #x = self.relu(self.fc1(x))
        features = self.fc1(x)
        out = self.fc2(features)
        return out, features