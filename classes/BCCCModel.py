"""
    author:housanqian
    create_date:2024/4/10
"""
from typing import Union
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import normalize


from classes.bias_corr_cc.squeezenet.Fire import Fire
from classes.bias_corr_cc.BCCC_module import (ChannelShuffle, GroupedDepthWiseSeparableConv2d, DeformConvSpatialAttention,
                                              ColorExtractor, ColorExtractorSelfAtt, Self_Attn,
                                              DeformConvSpatialAttentionStandard, DeformConvSpatialAttentionStandard1)

from auxiliary.settings import DEVICE





def correct_image_nolinear(img, ill):
    nonlinear_ill = torch.pow(ill, 1.0 / 2.2)
    correct = nonlinear_ill.unsqueeze(2).unsqueeze(3) * torch.sqrt(torch.Tensor([3])).to(DEVICE)
    correc_img = torch.div(img, correct + 1e-10)
    img_max = torch.max(torch.max(torch.max(correc_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    img_max = img_max.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    img_normalize = torch.div(correc_img, img_max)
    return img_normalize



class noBias(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 63, 63),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            ColorExtractor(126, 32, 258, kernel_size=5, is_sigmoid=True, is_deform=True),
            nn.BatchNorm2d(258),
            nn.ReLU(inplace=True),
            GroupedDepthWiseSeparableConv2d(258, 258, 3, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            ColorExtractor(258, 32, 384, kernel_size=3, is_sigmoid=True, is_deform=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            GroupedDepthWiseSeparableConv2d(384, 384, 3, 3),
            nn.ReLU(inplace=True),
            # ColorExtractor(384, 384),
            GroupedDepthWiseSeparableConv2d(384, 258, 3, 3),
            nn.BatchNorm2d(258),
            nn.ReLU(inplace=True),
            GroupedDepthWiseSeparableConv2d(258, 384, 3, 3),
            nn.ReLU(inplace=True),

        )

        self.final_convs = nn.Sequential(
            nn.MaxPool2d(2, 1, ceil_mode=True),
            nn.Conv2d(384, 192, kernel_size=(6, 1), stride=1, padding=(2, 0)),
            nn.Conv2d(192, 64, kernel_size=(1, 6), stride=1, padding=(0, 2)),
            # nn.Conv2d(384, 64, kernel_size=6, stride=1, padding=2),
            # nn.AvgPool2d(2, 1, ceil_mode=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.cc_weight = DeformConvSpatialAttentionStandard(in_channels=64, kernel_size=3,
                                                            is_sigmoid=False, is_deform=False)

        self.pre_convs = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Pretrained model annotations not loaded
        self.load_pretrained_except_final_convs()

    def load_pretrained_except_final_convs(self):

        pretrained_model_path = r"C:\Users\Administrator\Desktop\Bias-corrections-for-cc\classes\bias_corr_cc\BCCCPretrained\fold0\model.pth"

        pretrained_state = torch.load(pretrained_model_path)
        pretrained_dict = pretrained_state['model_state_dict']


        self.load_state_dict(pretrained_dict, strict=False)


    def forward(self, x: Tensor) -> Union[tuple, Tensor]:

        backbone_map = self.backbone(x)
        final_convs_map = self.final_convs(backbone_map)

        confidence = self.cc_weight(final_convs_map)
        rgb = self.pre_convs(final_convs_map)

        rgb = normalize(rgb, dim=1)

        pred = normalize(torch.sum(torch.sum(rgb * confidence, 2), 2), dim=1)

        return pred, rgb, confidence


class dobuleBias(nn.Module):
    def __init__(self, num_model=2):
        super(dobuleBias, self).__init__()
        self.submodel1 = noBias()
        self.submodel2 = noBias()
        self.submodel3 = noBias()

    def forward(self, x): 
        pred, rgb1, confidence1 = self.submodel1(x)
        correct_img1 = correct_image_nolinear(x, pred)

        b1, rgb2, confidence2 = self.submodel2(correct_img1)
        correct_img2 = correct_image_nolinear(x, torch.mul(pred, b1))

        b2, rgb3, confidence3 = self.submodel3(correct_img2)

        return pred, b1, b2, rgb3, confidence3


# 测试参数量
input_data = torch.randn(1, 3, 512, 512)  
model = noBias()
pred, rgb, confidence = model(input_data)
# print(output.shape)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
total_params = sum(p.numel() for p in model_parameters)
print(f"Total trainable parameters: {total_params}")
