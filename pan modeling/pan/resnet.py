import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1):
        super(Bottleneck, self).__init__()

        # block 1 : 1x1, inplanes
        #
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # block 2 : 3x3, inplanes, 여기만 stride가 1이 아니다.
        # 각 bottleneck 마다 어떻게 변화하는지 한번 살펴보자.
        # padding도 0이 아니다. 어떻게 변화하는지 살펴보자.
        # dilation도 0이 아니다.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=rate, dilation=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # block 3 : 1x1, inplanes * explansion
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

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

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # Conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Conv2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])

        # Conv3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # Conv4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        rates = [1, 2, 4]

        # Conv5
        self.layer4 = self._make_deeplabv3_layer(block, 512, layers[3], rates=rates, stride=1)  # stride 2 => stride 1

        # average pooling
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # fully_connected
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 초기화 방법
        # 모델의 모듈을 차례로 불러온다.
        for m in self.modules():
            # 모듈이 nn.Conv2d인 경우
            if isinstance(m, nn.Conv2d):
                '''
                모듈의 가중치를 kaiming he normal_로 초기화합니다.
                편차를 0으로 초기화합니다.
                '''
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                '''
                모듈의 가중치를 1, bias는 0으로 초기화 한다.
                '''
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # stride가 1이 아니거나 inplanes가 planes의 expansion배 만큼되지 않는 경우
        # downsample을 적용한다.
        # downsample이란 forward 시 f(x)+x의 residual을 구현할 경우 f(x)와 x의
        # 텐서 사이즈가 다른 경우에 사용한다.
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # block == bottleneck
        # local variable인가...
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        # 각 layer마다 들어가는 bottleneck의 갯수가 다르다.
        # 하단의 코드는 각 layer에 들어가는 block의 갯수만큼 block을 생성한다.
        # 아래의 코드는 좀 더 살펴보도록 하자.
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
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

        # 각 block마다 rate를 적용하는데 이는 atrous rate를 적용 [1,2,4]
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rates[i]))

        # 최종적으로 resnet block 하나를 반환한다.
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
