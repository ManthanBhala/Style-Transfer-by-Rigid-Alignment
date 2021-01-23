import torch
import torch.nn as nn

import copy
import os


feature_invertor_conv5_1 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,256,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,128,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,128,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,64,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,64,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,3,(3, 3)),
)


class Decoder(nn.Module):
    def __init__(self, depth, pretrained_path):
        super(Decoder, self).__init__()
        self.depth = depth
        if depth == 1:
            self.model = nn.Sequential(*copy.deepcopy(list(feature_invertor_conv5_1.children())[-2:]))
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'feature_invertor_conv1_1.pth')))
        elif depth == 2:
            self.model = nn.Sequential(*copy.deepcopy(list(feature_invertor_conv5_1.children())[-9:]))
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'feature_invertor_conv2_1.pth')))
        elif depth == 3:
            self.model = nn.Sequential(*copy.deepcopy(list(feature_invertor_conv5_1.children())[-16:]))
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'feature_invertor_conv3_1.pth')))
        elif depth == 4:
            self.model = nn.Sequential(*copy.deepcopy(list(feature_invertor_conv5_1.children())[-29:]))
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'feature_invertor_conv4_1.pth')))

    def forward(self, x):
        out = self.model(x)
        return out