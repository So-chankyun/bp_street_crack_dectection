import torch
import torch.nn as nn
from pan.modules.basic import Conv2dBn, Conv2dBnRelu
import torchvision

'''
Global Attention Upsample Module
'''


class GAUModule(nn.Module):
    def __init__(self, in_ch, out_ch):  #
        super(GAUModule, self).__init__()

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBn(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv2 = Conv2dBnRelu(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    # x: low level feature
    # y: high level feature
    def forward(self, x, y):
        h, w = x.size(2), x.size(3)
        y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)

        return y_up + z


'''
Feature Pyramid Attention Module
FPAModule1:
	downsample use maxpooling
'''


class FPAModule1(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(FPAModule1, self).__init__()

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        # midddle branch
        self.mid = nn.Sequential(
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=1, padding=3)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        b1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(b1)

        mid = self.mid(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)

        x2 = self.conv2(x2)
        x = x2 + x3
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)

        x1 = self.conv1(x1)
        x = x + x1
        x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)

        x = torch.mul(x, mid)
        x = x + b1
        return x


'''
Feature Pyramid Attention Module
FPAModule2:
	downsample use convolution with stride = 2
'''


class FPAModule2(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(FPAModule2, self).__init__()

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        # midddle branch
        self.mid = nn.Sequential(
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)

        self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)

        self.down3 = nn.Sequential(
            Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        b1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(b1)

        mid = self.mid(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)

        x2 = self.conv2(x2)
        x = x2 + x3
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)

        x1 = self.conv1(x1)
        x = x + x1
        x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)

        x = torch.mul(x, mid)
        x = x + b1
        return x


'''
papers:
	Pyramid Attention Networks
'''


class PAN(nn.Module):
    def __init__(self, backbone, pretrained=True, n_class=2):
        '''
        :param backbone: Bcakbone network
        '''
        super(PAN, self).__init__()

        if backbone.lower() == 'resnet34':
            encoder = torchvision.models.resnet34(pretrained)
            bottom_ch = 512
        elif backbone.lower() == 'resnet50':
            encoder = torchvision.models.resnet50(pretrained)
            bottom_ch = 2048
        elif backbone.lower() == 'resnet101':
            encoder = torchvision.models.resnet101(pretrained)
            bottom_ch = 2048
        elif backbone.lower() == 'resnet152':
            encoder = torchvision.models.resnet152(pretrained)
            bottom_ch = 2048
        else:
            raise NotImplementedError('{} Backbone not implement'.format(backbone))

        self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)  # 1/4
        self.conv2_x = encoder.layer1  # 1/4
        self.conv3_x = encoder.layer2  # 1/8
        self.conv4_x = encoder.layer3  # 1/16
        self.conv5_x = encoder.layer4  # 1/32

        self.fpa = FPAModule1(in_ch=bottom_ch, out_ch=n_class)

        self.gau3 = GAUModule(in_ch=bottom_ch // 2, out_ch=n_class)

        self.gau2 = GAUModule(in_ch=bottom_ch // 4, out_ch=n_class)

        self.gau1 = GAUModule(in_ch=bottom_ch // 8, out_ch=n_class)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x1 = self.conv1(x)
        x2 = self.conv2_x(x1)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        x5 = self.fpa(x5)  # 1/32
        x4 = self.gau3(x4, x5)  # 1/16
        x3 = self.gau2(x3, x4)  # 1/8
        x2 = self.gau1(x2, x3)  # 1/4

        out = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x2)

        return out

== == == =
import torch.nn as nn
from pan import resnet


class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        """Declare all needed layers."""
        super(ResNet50, self).__init__()
        self.model = resnet.resnet50(pretrained=pretrained)
        self.relu = self.model.relu  # Place a hook

        # 이 부분이 어떤 것을 의미하는지 잘 모르겠다....
        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

    def forward(self, x):
        feature_map = []

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # pretrained라서 그냥 blocks 불러와도 바로 쓸 수 있는건가...?
        # 맞다면 첫 layer에서 정의된 값을 가지고 해당 layer의 bottleneck들로 feature_map을 생성함.
        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)

        # 코드는 이해가 안됨. 하지만 여기서 하는 연산은 feature map의 class를 부여하는 것으로 보임.
        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        return feature_map, out


class Classifier(nn.Module):
    def __init__(self, in_features=2048, num_class=20):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, num_class)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        return x


class FPA(nn.Module):
    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels / 4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3,
                                   bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1,
                                                  bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1,
                                                  bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1,
                                                  bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)  # 이 부분이 실행이 안된다.

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2

        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

        out = self.relu(x_master + x_gpb)

        return out


class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1,
                                                    bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        # batch size ,channels, height, weight
        # view, reshape와 같은 역할을 한다. (bxCx1x1) 형태로 변환을 진행해준다.
        # high-feature map에 average pooling을 적용한 후에 reshape 진행.
        # 그리고 1x1 kernel로 conv해주고 BN, ReLU 진행.
        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)  # 이걸 실시함으로써 high-low 만큼의 정보 손실이 발생하는 것인가...
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out


class PAN(nn.Module):
    def __init__(self, blocks=[]):
        """
        :param blocks: Blocks of the network with reverse sequential.
        """
        super(PAN, self).__init__()
        channels_blocks = []

        # 어떤 작업인지 이해가 안됨.
        for i, block in enumerate(blocks):
            channels_blocks.append(list(list(block.children())[2].children())[4].weight.shape[0])

        self.fpa = FPA(channels=channels_blocks[0])
        # channels_high = channels_blocks[0]
        # for i, channels_low in enumerate(channels_blocks[1:]):
        #     self.gau.append(GAU(channels_high, channels_low))
        #     channels_high = channels_low
        self.gau_block1 = GAU(channels_blocks[0], channels_blocks[1], upsample=False)
        self.gau_block2 = GAU(channels_blocks[1], channels_blocks[2])
        self.gau_block3 = GAU(channels_blocks[2], channels_blocks[3])
        self.gau = [self.gau_block1, self.gau_block2, self.gau_block3]

        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms=[]):
        """
        :param fms: Feature maps of forward propagation in the network with reverse sequential. shape:[b, c, h, w]
        :return: fm_high. [b, 256, h, w]
        """
        for i, fm_low in enumerate(fms):
            if i == 0:
                fm_high = self.fpa(fm_low)
            else:
                fm_high = self.gau[int(i - 1)](fm_high, fm_low)

        return fm_high


class Mask_Classifier(nn.Module):
    def __init__(self, in_features=512, num_class=1):
        super(Mask_Classifier, self).__init__()
        self.mask_conv = nn.ConvTranspose2d(in_features, num_class, kernel_size=6, stride=4, padding=1)

    def forward(self, x):
        # 먼저 upsampling해주고... 걍 한번에 진행하자.
        x = self.mask_conv(x)
        return x