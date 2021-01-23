import torch
import torch.nn as nn

from models.encoder import *
from models.decoder import *
from rigid_alignment import *


class MultiLevelNetwork(nn.Module):
    def __init__(self, device, pretrained_path, alpha=0.5, beta=0):
        super(MultiLevelNetwork, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        
        self.e1 = Encoder(1, pretrained_path)
        self.e2 = Encoder(2, pretrained_path)
        self.e3 = Encoder(3, pretrained_path)
        self.e4 = Encoder(4, pretrained_path)
        self.encoders = [ self.e4, self.e3, self.e2, self.e1]
        
        self.d1 = Decoder(1, pretrained_path)
        self.d2 = Decoder(2, pretrained_path)
        self.d3 = Decoder(3, pretrained_path)
        self.d4 = Decoder(4, pretrained_path)
        self.decoders = [self.d4, self.d3, self.d2, self.d1]

    def forward(self, content_img, style_img, additional_style_flag=False, style_img1=None):
        if additional_style_flag:
            content_img = stylize_ra(0, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha,
                                     beta=self.beta, style1=style_img1)
            content_img = stylize_mm(1, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha,
                                     beta=self.beta, style1=style_img1)
            content_img = stylize_mm(2, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha,
                                     beta=self.beta, style1=style_img1)
            content_img = stylize_mm(3, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha,
                                     beta=self.beta, style1=style_img1)
        else:
            content_img = stylize_ra(0, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha)
            content_img = stylize_mm(1, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha)
            content_img = stylize_mm(2, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha)
            content_img = stylize_mm(3, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha)
        return content_img