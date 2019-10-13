#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Define your architecture here.
### Feel free to use as many code cells as needed.
import torchvision
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn
import torch.nn.functional as F


class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        
        # Conv1
        self.conv1_1 = nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=True)   # This two conv layers equal to one conv layer with
        self.conv1_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=True)  # kernel_size = 5, filters = 16, but with less parameters
        self.bn1 = nn.BatchNorm2d(16)                                      
        self.pool1 = nn.MaxPool2d(2, 2)                                                                      
        
        # Conv2
        self.conv2_1 = nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=True)  
        self.conv2_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True)  
        self.bn2 = nn.BatchNorm2d(32)                                        
        self.pool2 = nn.MaxPool2d(2, 2)                                     
        
        # Conv3
        self.conv3_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True)  
        self.conv3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)  
        self.bn3 = nn.BatchNorm2d(64)                                        
        self.pool3 = nn.MaxPool2d(2, 2)                                      
        
        # FC
        self.fc4 = nn.Linear(64 * 4 * 4, 512)
        self.fc5 = nn.Linear(512, 43)
        
        self.init_conv2d()

    def forward(self, images):
        # x: input size (N, 1, 32, 32)
        out = F.relu(self.conv1_1(images))    # (N, 16, 32, 32)
        out = F.relu(self.conv1_2(out))       # (N, 16, 32, 32)
        out = self.bn1(out)
        out = self.pool1(out)                 # (N, 16, 16, 16)
        #print(out.size())
        
        out = F.relu(self.conv2_1(out))       # (N, 32, 16, 16)
        out = F.relu(self.conv2_2(out))       # (N, 32, 16, 16)
        out = self.bn2(out)
        out = self.pool2(out)                 # (N, 32, 8, 8)
        #print(out.size())
        
        out = F.relu(self.conv3_1(out))       # (N, 64, 8, 8)
        out = F.relu(self.conv3_2(out))       # (N, 64, 8, 8)
        out = self.bn3(out)
        out = self.pool3(out)                 # (N, 64, 4, 4)
        #print(out.size())
        
        out = out.view(-1, 64*4*4)            # (N, 1024)
        out = nn.Dropout(0.5)(out)
        
        out = F.relu(self.fc4(out))           # (N, 512)
        out = nn.Dropout(0.8)(out)
        out = self.fc5(out)                   # (N, 43)
        
        return out
    
    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight)
                nn.init.constant_(c.bias, 0)





