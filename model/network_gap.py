import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input image size : 28x28

        # Convolution Layer 1
        #self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #input -? OUtput? RF
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1), # Input Channels - 1, Output Channels - 12
            nn.BatchNorm2d(12),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        
        # Convolution Layer 2
        #self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1), # Input Channels - 12, Output Channels - 24
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        
        # Transition Layer 1 - MaxPool and 1x1 conv
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),     # Input Channels - 24, Output Channels - 24
            nn.Conv2d(32, 12, kernel_size=1, padding=0),   # Input Channels - 24, Output Channels - 12
        )

        # Convolution Layer 3
        #self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, padding=1), # Input Channels - 12, Output Channels - 16
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        # Convolution Layer 4
        #self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Input Channels - 12, Output Channels - 16
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        # Transition Layer 2 - MaxPool and 1x1 conv
        self.transition2 = nn.Sequential(
            nn.MaxPool2d(2, 2),     # Input Channels - 24, Output Channels - 24
            nn.Conv2d(32, 12, kernel_size=1, padding=0),   # Input Channels - 24, Output Channels - 12
        )
        
        # Convolution Layer 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1), # Input Channels - 12, Output Channels - 16
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        # Convolution Layer 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=3, padding=0), # Input Channels - 12, Output Channels - 16
            nn.BatchNorm2d(10),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        
        # Convolution Layer 7
        self.conv7 = nn.Sequential(
            nn.Conv2d(12, 10, kernel_size=3, padding=0), # Input Channels - 12, Output Channels - 16
            nn.ReLU()
        )
        
        #self.fc = nn.Sequential(
        #    nn.Linear(90, 10)
        #)
        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transition1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.transition2(x)

        x = self.conv5(x)
        x = self.conv6(x)

        #x = self.conv7(x)

        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        x = self.gap(x)
        x = x.view(-1, 10) 
        return F.log_softmax(x, dim=1)
