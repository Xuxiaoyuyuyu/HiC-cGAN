# -*- coding: utf-8 -*-
from torch import nn

# +
class ConvolutionalBlock(nn.Module):
    """
    卷积模块,由卷积层, BN归一化层, 激活层构成.
    """

#     w_init = tf.random_normal_initializer(stddev=0.02)
#     b_init = None  # tf.constant_initializer(value=0.0)
#     g_init = tf.random_normal_initializer(1., 0.02)

#     w_init = nn.init.normal_(mean=0, std=0.02)
#     b_init = None
#     g_init = nn.init.normal_(mean=1, std=0.02)
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :参数 in_channels: 输入通道数
        :参数 out_channels: 输出通道数
        :参数 kernel_size: 核大小
        :参数 stride: 步长
        :参数 batch_norm: 是否包含BN层
        :参数 activation: 激活层类型; 如果没有则为None
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh', 'relu'}

        # 层列表
        layers = list()

        # 1个卷积层
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=True)
        conv.weight = nn.init.normal_(conv.weight, mean=0, std=0.02)
        conv.bias = None
        layers.append(conv)

        # 1个激活层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'relu':
            layers.append(nn.ReLU())

        # 1个BN归一化层
        if batch_norm is True:
            BN = nn.BatchNorm2d(num_features=out_channels)
#             BN.gamma = nn.init.normal_(BN.gamma, mean=1, std=0.02)
            layers.append(BN)

        # 合并层
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        output = self.conv_block(input)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super(ResidualBlock, self).__init__()

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)
        return output


class Generator(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64, n_blocks=5):
        super(Generator, self).__init__()

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=1, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=False, activation='relu')

        # 一系列残差模块, 每个残差模块包含一个跳连
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

        # 最后3个卷积模块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=128, kernel_size=kernel_size,
                                              batch_norm=False, activation=None)
        self.conv_block4 = ConvolutionalBlock(in_channels=128, out_channels=256, kernel_size=kernel_size,
                                              batch_norm=False, activation=None)
        self.conv_block5 = ConvolutionalBlock(in_channels=256, out_channels=1, kernel_size=1,
                                              batch_norm=False, activation='tanh')

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)  # (batch_size, 1, 40, 40)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = output + residual
        output = self.conv_block3(output)
        output = self.conv_block4(output)
        sr_imgs = self.conv_block5(output)

        return sr_imgs


class Discriminator(nn.Module):
    
    def __init__(self, kernel_size=3, n_channels=64):
        super(Discriminator, self).__init__()

        # 卷积系列
        layers = list()
        layers.append(ConvolutionalBlock(in_channels=1, out_channels=n_channels, kernel_size=kernel_size, 
                                         stride=1, batch_norm=False, activation='LeakyReLu'))
        layers.append(ConvolutionalBlock(in_channels=64, out_channels=n_channels, kernel_size=kernel_size,
                                         stride=2, batch_norm=True, activation='LeakyReLu'))
        layers.append(ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                         stride=1, batch_norm=True, activation='LeakyReLu'))
        layers.append(ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                         stride=2, batch_norm=True, activation='LeakyReLu'))
        layers.append(ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                         stride=1, batch_norm=True, activation='LeakyReLu'))
        layers.append(ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                         stride=2, batch_norm=True, activation='LeakyReLu'))
        layers.append(ConvolutionalBlock(in_channels=n_channels, out_channels=128, kernel_size=kernel_size,
                                         stride=1, batch_norm=True, activation='LeakyReLu'))

        # 固定输出大小
        # layers.append(nn.AdaptiveAvgPool2d((6, 6)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(128*5*5, 512))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(512, 1))
        #layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, imgs):
        result = self.net(imgs)
        return result
