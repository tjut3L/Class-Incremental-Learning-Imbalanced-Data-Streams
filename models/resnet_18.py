"""This is the slimmed ResNet as used by Lopez et al. in the GEM paper."""
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
import models.modified_linear as modified_linear

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(
        #             in_planes,
        #             self.expansion * planes,
        #             kernel_size=1,
        #             stride=stride,
        #             bias=False,
        #         ),
        #         nn.BatchNorm2d(self.expansion * planes),
        #     )

    def forward(self, x):
        residual = x
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if not self.last:
            out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes, nf):
        self.in_planes = nf
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 3, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, last_phase=True)
        # # self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.linear = modified_linear.CosineLinear(nf * 8 * block.expansion, num_classes)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, last_phase=True)
        self.linear = modified_linear.CosineLinear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.in_planes, planes))
            layers.append(block(self.in_planes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        # out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, x.size()[-1], x.size()[-1]))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        representation = out
        out = self.linear(out)
        return out, representation

def SlimResNet18(n_classes, nf=64):
    """Slimmed ResNet18."""
    return ResNet(BasicBlock, [2, 2, 2, 2], n_classes, nf)


__all__ = ["SlimResNet18"]
