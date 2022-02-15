import torch.nn as nn
import torch.nn.functional as F

class Hswish(nn.Module):
    def forward(self, x):
        swish = F.relu6(x + 3 , inplace = True)
        return x* swish/6.

class Hsigmoid(nn.Module):

    def forward(self, x):
        return F.relu6(x + 3, inplace = True)/6.

class SElayer(nn.Module):
    def __init__(self, inplanes, ratio = 0.25):
        super(SElayer, self).__init__()
        hidden_dim = int(inplanes*ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(inplanes, hidden_dim, bias = False)
        self.fc2 = nn.Linear(hidden_dim, inplanes, bias = False)
        self.activate = Hsigmoid()
        

    def forward(self, x):
        out = self.avg_pool(x).view(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.activate(out)
        out = out.unsqueeze(2).unsqueeze(3)
        out = x * out.expand_as(x)

        return out

class conv_set(nn.Module):
    """docstring for conv_set"""
    def __init__(self, in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1):
        super(conv_set, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_ch),
            Hswish()
            )
    def forward(self, x):
        out = self.conv(x)
        return out

class Bottleneck(nn.Module):

    def __init__(self, in_ch, out_ch, stride, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_set(in_ch, out_ch, stride = stride)
        self.conv2 = conv_set(out_ch, out_ch)
        self.se = SElayer(out_ch)
        if in_ch != out_ch:
            self.downsample = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_ch))
        else:
            self.downsample = lambda x: x

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)

        out += residual

        return out

class ResXnet(nn.Module):
    def __init__(self, block, size):
        super(ResXnet, self).__init__()
        self.conv = conv_set(1, 32)
        self.in_ch = 32
        self.layer1 = self.make_layer(block, 64, size, stride = 2)
        self.layer2 = self.make_layer(block, 128, size, stride = 2)
        self.layer3 = self.make_layer(block, 256, size, stride = 2)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_ch, planes, stride))
            self.in_ch = planes

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return out



def ResXnet56():
    model = ResXnet(Bottleneck, 9)

    return model



if __name__ == "__main__":
    import torch
    model = ResXnet56()
    input = torch.rand(1,1, 400,400)
    output = model(input)
    print(output.size())
    