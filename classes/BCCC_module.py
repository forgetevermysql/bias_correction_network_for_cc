"""
    author:housanqian
    create_date:2024/4/10
"""
from typing import Union

import torch
from torch import nn

from classes.bias_corr_cc.DeformConv2d import DeformConv2d


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        assert channels_per_group * self.groups == num_channels, "通道清洗：通道数应该可以被groups整除"

        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        x = x.transpose(1, 2).contiguous()

        x = x.view(batch_size, -1, height, width)

        return x



class GroupedDepthWiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=1, padding=1, is_shuffle=True):
        super(GroupedDepthWiseSeparableConv2d, self).__init__()
        self.is_shuffle = is_shuffle

        assert in_channels % groups == 0, "in_channels should be divisible by groups"
        assert out_channels % groups == 0, "out_channels should be divisible by groups"

        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups

        self.grouped_depthwise_convs = nn.ModuleList([
            nn.Conv2d(self.in_channels_per_group, self.in_channels_per_group, kernel_size, stride, padding,
                      groups=self.in_channels_per_group)
            for _ in range(groups)
        ])

        self.grouped_pointwise_convs = nn.ModuleList([
            nn.Conv2d(self.in_channels_per_group, self.out_channels_per_group, 1)
            for _ in range(groups)
        ])

    def forward(self, x):
        if self.is_shuffle:
            shuffle = ChannelShuffle(3)
            x = shuffle(x)

        input_groups = torch.split(x, self.in_channels_per_group, dim=1)

        output_groups = []
        for depthwise_conv, pointwise_conv, group in zip(self.grouped_depthwise_convs, self.grouped_pointwise_convs,
                                                         input_groups):
            out = depthwise_conv(group)
            out = pointwise_conv(out)
            output_groups.append(out)

        return torch.cat(output_groups, dim=1)


# conv = GroupedDepthWiseSeparableConv2d(in_channels=126, out_channels=510, kernel_size=3, groups=3, padding=1)
# input_tensor = torch.randn(16, 126, 32, 32)
# output = conv(input_tensor)
# total_params = sum(p.numel() for p in conv.parameters())
# print("Total number of parameters:", total_params)
# print(output.shape)  # 输出应为torch.Size([1, 16, 32, 32])


class DeformConvSpatialAttention(nn.Module):
    def __init__(self, in_channels, is_sigmoid=True):
        super(DeformConvSpatialAttention, self).__init__()

        self.is_sigmoid = is_sigmoid

        self.channel_reduce = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)

        # self.feature_extractor = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.feature_extractor = DeformConv2d(2, 1)

    def forward(self, x):


        x = self.channel_reduce(x)

        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)

        max_pooled = max_pooled[:, :, :x.size(2), :x.size(3)]
        avg_pooled = avg_pooled[:, :, :x.size(2), :x.size(3)]

        pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # 维度: [B, 2, H, W]

        attention = self.feature_extractor(pooled)
        if self.is_sigmoid:

            attention = torch.sigmoid(attention)

        return attention



# modified_attention_module = DeformConvSpatialAttention(in_channels=63)
# modified_attention_map = modified_attention_module(input_tensor)
# print("attention map:", modified_attention_map.shape)  # 返回输出的尺寸，应该与输入的空间尺寸相同，但通道数为1
# total_params = sum(p.numel() for p in modified_attention_module.parameters())
# print("Total number of parameters:", total_params)


class DeformConvSpatialAttentionStandard(nn.Module):
    def __init__(self, in_channels, kernel_size=3, is_sigmoid=True, is_deform=True):
        super(DeformConvSpatialAttentionStandard, self).__init__()

        self.kernel_size = kernel_size
        self.is_sigmoid = is_sigmoid
        self.is_deform = is_deform

        if self.is_deform:
            self.feature_extractor = DeformConv2d(2, 1)
        else:
            self.feature_extractor = nn.Conv2d(2, 1, kernel_size=self.kernel_size, stride=1,
                                               padding=self.kernel_size // 2)

    def forward(self, x):

        max_pooled, _ = torch.max(x, dim=1, keepdim=True)
        avg_pooled = torch.mean(x, dim=1, keepdim=True)

        pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # 维度: [B, 2, H, W]

        attention = self.feature_extractor(pooled)
        if self.is_sigmoid:

            attention = torch.sigmoid(attention)

        return attention


class ColorExtractor(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, is_sigmoid=True, is_deform=True):
        super(ColorExtractor, self).__init__()

        self.kernel_size = kernel_size
        self.is_sigmoid = is_sigmoid
        self.is_deform = is_deform

        self.in_channels = in_channels

        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

        self.branch1_conv1x5 = nn.Conv2d(mid_channels, out_channels//2, kernel_size=(1, 5), padding=(0, 2))
        self.branch1_conv5x1 = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=(5, 1), padding=(2, 0))

        self.branch2_conv3x3 = nn.Conv2d(mid_channels, out_channels//2, kernel_size=3, padding=1)


        self.dcsa = DeformConvSpatialAttentionStandard(self.in_channels, kernel_size=self.kernel_size,
                                                      is_sigmoid=self.is_sigmoid, is_deform=True)


    def forward(self, x):

        cc_weight = self.dcsa(x)
        x = x * cc_weight
        x = self.conv1x1(x)

        # Branch 1
        out_branch1 = self.branch1_conv5x1(self.branch1_conv1x5(x))

        # Branch 2
        out_branch2 = self.branch2_conv3x3(x)

        # concat
        out = torch.cat([out_branch1, out_branch2], dim=1)

        return out


# self-attention
class Self_Attn(nn.Module):


    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):

        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class ColorExtractorSelfAtt(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ColorExtractorSelfAtt, self).__init__()

        self.in_channels = in_channels
        # 1*1卷积，降维
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

        # Branch 1: 1*5卷积接5*1卷积
        # self.branch1_conv1x5 = nn.Conv2d(mid_channels, out_channels//2, kernel_size=(1, 5), padding=(0, 2))
        # self.branch1_conv5x1 = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=(5, 1), padding=(2, 0))
        # self.branch1_conv1x1 = nn.Conv2d(mid_channels, out_channels//2, kernel_size=(5, 1), padding=(2, 0))
        # self.branch1_conv5x1 = nn.Conv2d(out_channels, out_channels//2, kernel_size=5, padding=2)
        self.dcsa = DeformConv2d(mid_channels, out_channels//2, kernel_size=3, padding=1)

        # Branch 2: 3*3 卷积
        self.branch2_conv3x3 = nn.Conv2d(mid_channels, out_channels//2, kernel_size=5, padding=2)

        self.self_attention = Self_Attn(self.in_channels)

    def forward(self, x):
        # 自注意力
        x, _ = self.self_attention(x)
        x = self.conv1x1(x)

        # Branch 1
        out_branch1 = self.dcsa(x)

        # Branch 2
        out_branch2 = self.branch2_conv3x3(x)

        # concat
        out = torch.cat([out_branch1, out_branch2], dim=1)

        return out


# cc_extractor = ColorExtractorSelfAtt(in_channels=126, mid_channels=32, out_channels=258)
# input_tensor = torch.randn(16, 126, 32, 32)
# print("input_tensor_shape", input_tensor.shape)
# cc_extractor_map = cc_extractor(input_tensor)
# print("attention map:", cc_extractor_map.shape)  # 返回输出的尺寸，应该与输入的空间尺寸相同，但通道数为1
# total_params = sum(p.numel() for p in cc_extractor.parameters())
# print("Total number of parameters in cc_extractor:", total_params)


class DeformConvSpatialAttentionStandard1(nn.Module):
    def __init__(self, in_channels, kernel_size=3, is_sigmoid=True, is_deform=True):
        super(DeformConvSpatialAttentionStandard1, self).__init__()

        self.kernel_size = kernel_size
        self.is_sigmoid = is_sigmoid
        self.is_deform = is_deform

        # 使用卷积层从堆叠的池化结果中提取特征
        if self.is_deform:
            self.feature_extractor = DeformConv2d(2, 1)
        else:
            self.feature_extractor = nn.Conv2d(2, 1, kernel_size=self.kernel_size, stride=1,
                                               padding=self.kernel_size // 2)

    def forward(self, x):
        # 通道减少

        # print("feature_extractor weight device:", self.feature_extractor.weight.device)

        max_pooled, _ = torch.max(x, dim=1, keepdim=True)
        avg_pooled = torch.mean(x, dim=1, keepdim=True)

        # 在通道维度上堆叠池化结果
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # 维度: [B, 2, H, W]

        # 使用卷积层提取特征
        attention = self.feature_extractor(pooled)
        if self.is_sigmoid:
            # 使用sigmoid函数确保注意力权重在[0, 1]范围内
            attention = torch.sigmoid(attention)
        attention = attention * x

        return attention

