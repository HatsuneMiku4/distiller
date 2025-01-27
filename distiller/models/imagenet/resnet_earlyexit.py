import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import BasicBlock


__all__ = ['resnet50_earlyexit']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResNetEarlyExit(models.ResNet):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNetEarlyExit, self).__init__(block, layers, num_classes)

        # Define early exit layers
        self.conv1_exit0 = nn.Conv2d(256, 50, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv2_exit0 = nn.Conv2d(50, 12, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv1_exit1 = nn.Conv2d(512, 12, kernel_size=7, stride=2, padding=3, bias=True)
        self.fc_exit0 = nn.Linear(147 * block.expansion, num_classes)
        self.fc_exit1 = nn.Linear(192 * block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # Add early exit layers
        exit0 = self.avgpool(x)
        exit0 = self.conv1_exit0(exit0)
        exit0 = self.conv2_exit0(exit0)
        exit0 = self.avgpool(exit0)
        exit0 = exit0.view(exit0.size(0), -1)
        exit0 = self.fc_exit0(exit0)

        x = self.layer2(x)

        # Add early exit layers
        exit1 = self.conv1_exit1(x)
        exit1 = self.avgpool(exit1)
        exit1 = exit1.view(exit1.size(0), -1)
        exit1 = self.fc_exit1(exit1)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # return a list of probabilities
        output = [exit0, exit1, x]
        return output


def resnet50_earlyexit(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNetEarlyExit(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
