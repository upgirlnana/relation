# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
from torch import nn
from models.models_utils.rga_modules import RGA_Module
from models.models_utils.part_rga_modules import Part_RGA_Module


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1)
        init.constant(m.bias.data, 0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []

        add_block += [nn.Conv2d(input_dim, num_bottleneck, kernel_size=1, bias=False)]
        add_block += [nn.BatchNorm2d(num_bottleneck)]
        if relu:
            #add_block += [nn.LeakyReLU(0.1)]
            add_block += [nn.ReLU(inplace=True)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Relation_with_region(nn.Module):
    def __init__(self, class_num,last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3],
                spa_on=True, cha_on=True, s_ratio=8, c_ratio=8, d_ratio=8, height=256, width=128,):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

        self.rga_att1 = RGA_Module(256, (height // 4) * (width // 4), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
        self.rga_att2 = RGA_Module(512, (height // 8) * (width // 8), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
        self.rga_att3 = RGA_Module(1024, (height // 16) * (width // 16), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
        self.rga_att4 = RGA_Module(2048, (height // 16) * (width // 16), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
        self.region_att = Part_RGA_Module(2048, (6) * (1), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=1, down_ratio=d_ratio)
        self.classifiers = nn.ModuleList()
        for i in range(self.part):
            self.classifiers.append(ClassBlock(2048, class_num, True, 256))
        for i in range(self.part):
            self.feature.append(self.classifiers[i].add_block)
            self.cls.append(self.classifiers[i].classifier)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.rga_att1(x)

        x = self.layer2(x)
        x = self.rga_att2(x)

        x = self.layer3(x)
        x = self.rga_att3(x)

        x = self.layer4(x)
        partfeature=self.region_att(x)
        part = {}
        predict = {}
        feature = {}
        f = []
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = partfeature[:, :, i, :]
            part[i] = torch.unsqueeze(part[i], 3)
            # print part[i].shape
            feature[i] = self.feature[i](part[i])
            # print(feature[i].size())
            f.append(feature[i])
            # predict[i] = self.cls[i](feature[i])
            predict[i] = self.classifiers[i](part[i])

        # for i in range(self.part):

        x = torch.cat(f, 2)

        feat = x.view(x.size(0), x.size(1), x.size(2))
        cls = []
        for i in range(self.part):
            cls.append(predict[i])
        # return cls, feat


        x = self.rga_att4(x)

        return x,feat,cls

    def load_param(self, model_path):
        param_dict = torch.load(model_path)

        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

