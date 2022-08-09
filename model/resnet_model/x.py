import torch
import torch.nn as nn

Layers = [3, 4, 6, 3]
class Block(nn.Module):# nn.model 在类中实现网络各层的定义及前项计算和反向传播机制
    def __init__(self, in_channels, filters, stride=1, is_1x1conv=False):
        # in_channels 通道数
        super(Block, self).__init__()
        filter1, filter2, filter3 = filters  # 各层卷积核个数
        self.is_1x1conv = is_1x1conv # 直接将浅层的特征图仅仅经历一次卷积的捷径，正常情况下应该是三次卷积 跳跃连接
        self.relu = nn.ReLU(inplace=True)# 原地覆盖
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride,bias=False),
            nn.BatchNorm2d(filter1), # BN层 相当于加快网络的训练和收敛的速度
            # 控制梯度爆炸防止梯度消失  防止过拟合
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1,  bias=False),
            nn.BatchNorm2d(filter3),
            # 没有加上Relu（）函数，主要是这里需要判断这个板块是否激活了self.shortcut,
            # 只有加上这个之后才能一起Relu。
        )
        # 这段代码就是特征图捷径，浅层特征图就经历一次卷积直接与进行三次卷积之后的特征图相加
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride,  bias=False),
                nn.BatchNorm2d(filter3)
            )
    def forward(self, x):
        x_shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.is_1x1conv:
            x_shortcut = self.shortcut(x_shortcut)
        x = x + x_shortcut
        x = self.relu(x)
        return x


class Resnet50(nn.Module):

    def __init__(self):
        super(Resnet50,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = self._make_layer(64, (64, 64, 256), Layers[0])
        self.conv3 = self._make_layer(256, (128, 128, 512), Layers[1], 2)
        self.conv4 = self._make_layer(512, (256, 256, 1024), Layers[2], 2)
        self.conv5 = self._make_layer(1024, (512, 512, 2048), Layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 自适应平均池化


        self.fc = nn.Sequential(
            nn.Linear(2048, 10) # 全连接
        )
    def forward(self, input):
        x = self.conv1(input)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 扁平化
        x = self.fc(x)
        return x
    def _make_layer(self, in_channels, filters, blocks, stride=1): # blocks 残差块的个数
        layers = [] # 记录每次残差块的数据
        block_1 = Block(in_channels, filters, stride=stride, is_1x1conv=True)
        layers.append(block_1)
        print(len(layers))
        for i in range(1, blocks):
            print(filters[2])
            layers.append(Block(filters[2], filters, stride=1, is_1x1conv=False))

        return nn.Sequential(*layers) # *args传的是元组  **kwargs传的是字典

