import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def get_model(model_name, num_classes=2):
#     """
#     Returns the model architecture for the provided model_name.

#     Args:
#         model_name: Name of the model architecture to be returned.
#                     Options: ['resnet34']
#         num_classes: Number of classes for the final layer
#     Returns:
#         model: nn.Module object representing the model architecture.
#     """
#     if model_name == "resnet34":
#         model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=2)
#     else:
#         assert False, f'Unknown network architecture "{model_name}"'
#     return model


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet(nn.Module):
    def __init__(self, in_channels, resblock_cls, outputs=1):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            resblock_cls(64, 64, downsample=False),
            resblock_cls(64, 64, downsample=False),
        )

        self.layer2 = nn.Sequential(
            resblock_cls(64, 128, downsample=True),
            resblock_cls(128, 128, downsample=False),
        )

        self.layer3 = nn.Sequential(
            resblock_cls(128, 256, downsample=True),
            resblock_cls(256, 256, downsample=False),
        )

        self.layer4 = nn.Sequential(
            resblock_cls(256, 512, downsample=True),
            resblock_cls(512, 512, downsample=False),
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, 1)
        input = self.fc(input)
        return input


# class ResidualBlock(nn.Module):
#     """
#     Reusable convolutional layer with batch normalization and ReLU
#     """

#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=3,
#                 stride=stride,
#                 padding=1,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 out_channels, out_channels, kernel_size=3, stride=1, padding=1
#             ),
#             nn.BatchNorm2d(out_channels),
#         )
#         self.downsample = downsample
#         self.relu = nn.ReLU()
#         self.out_channels = out_channels

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=2):
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3)
#         )
#         # self.net = nn.Sequential(
#         #     nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3),  # conv1
#         #     nn.BatchNorm2d(64),  # conv1
#         #     nn.ReLU(),  # conv1
#         #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         #     self.make_layer(block, 64, layers[0], stride=1),
#         #     self.make_layer(block, 128, layers[1], stride=2),
#         #     self.make_layer(block, 256, layers[2], stride=2),
#         #     self.make_layer(block, 512, layers[3], stride=2),
#         #     nn.AvgPool2d(7, stride=1),
#         # )
#         self.fc = nn.Linear(512, num_classes)

#     def make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes),
#             )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.net(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x
