# Import Dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

# Unet
class UNet(nn.Module):
    def __init__(self, num_filters, num_colours, num_in_channels, kernel=3):
        super(UNet, self).__init__()
        # Calculate padding
        padding = kernel // 2
        # Model
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_in_channels, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(num_filters*2 + 64, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU()
            )

        self.layer4 = nn.Sequential(
            nn.Conv2d(num_filters*2+num_filters*2, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),            
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(num_filters+num_filters, num_colours, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU(),
            )
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(num_colours+num_in_channels, num_colours, kernel_size=kernel, padding=padding),
        )  


    def forward(self, x, feature_tensor): 
        self.o1 = self.layer1(x)
        self.o2 = self.layer2(self.o1)
        self.o3 = self.layer3(torch.cat((self.o2,feature_tensor),1))
        self.o4 = self.layer4(torch.cat((self.o3, self.o2),1))
        self.o5 = self.layer5(torch.cat((self.o4, self.o1),1))
        self.o6 = self.layer6(torch.cat((self.o5,x),1))
        return self.o6

