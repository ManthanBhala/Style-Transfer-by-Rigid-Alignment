import torch
import torch.nn as nn

import copy
import os


vgg_normalised_conv5_1 = nn.Sequential(
    nn.Conv2d(3,3,(1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3,64,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,64,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,128,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,128,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
)


class Encoder(nn.Module):
    def __init__(self, depth, pretrained_path):
        super(Encoder, self).__init__()
        self.depth = depth
        if depth == 1:
            self.model = vgg_normalised_conv5_1[:4]
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'vgg_normalised_conv1_1.pth')))
        elif depth == 2:
            self.model = vgg_normalised_conv5_1[:11]
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'vgg_normalised_conv2_1.pth')))
        elif depth == 3:
            self.model = vgg_normalised_conv5_1[:18]
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'vgg_normalised_conv3_1.pth')))
        elif depth == 4:
            self.model = vgg_normalised_conv5_1[:31]
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'vgg_normalised_conv4_1.pth')))

    def forward(self, x):
        out = self.model(x)
        return out