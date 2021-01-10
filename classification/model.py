# Import Dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

# Submodule model 
class Submodel_1(nn.Module):
    def __init__(self,model):
        super(Submodel_1, self).__init__()
        image_modules = list(model.children())[:-5] #all layer expect last five layers
        self.modelA = nn.Sequential(*image_modules)
        
    def forward(self, image):
        a = self.modelA(image)
        x = F.sigmoid(a)
        return x

# classifier
class Classifier(nn.Module):
    def __init__(self,num_in_features):
        super(Classifier, self).__init__()
        self.name = 'classifier'
        self.num_in_features = num_in_features
        self.fc1 = nn.Linear(self.num_in_features, 5000)
        self.fc2 = nn.Linear(5000, 320)
        self.fc3 = nn.Linear(320, 1)

    def forward(self, x):
        x = x.view(-1, self.num_in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        X = F.sigmoid(x)
        x = x.squeeze(1)
        return x
