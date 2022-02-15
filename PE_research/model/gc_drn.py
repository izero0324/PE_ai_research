import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F 

def conv3x3(inplanes, planes, stride = 1, dilation = 1):
    return nn.Conv2d(inplanes, planes, kernal_size = 3, stride = stride,
                     padding = padding, bias = False, dilation = dilation)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))

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
        out = self.avg_pool(x).view(x.size(0), -1) #???
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.activate(out)
        out = out.unsqueeze(2).unsqueeze(3)
        out = x * out.expand_as(x)

        return out

class GClayer(nn.Module):
    def __init__(self, inplanes, ratio = 0.25):
        super(GClayer, self).__init__()
        hidden_dim = int(inplanes*ratio)
        self.softmax = nn.Softmax(dim=2)
        self.conv1 = nn.Conv2d(inplanes, 1, 1)
        self.conv2 = nn.Conv2d(inplanes, hidden_dim, 1)
        self.conv3 = nn.Conv2d(hidden_dim, inplanes, 1)
        self.ln = nn.LayerNorm([hidden_dim, 1, 1])
        self.act = Mish()

    def forward(self, x):
        batch, channel, height, width = x.size()
        context = x.view(batch, channel, height*width).unsqueeze(2)
        #print(context.shape)
        mask = self.conv1(context)
        mask = mask.view(batch, 1, height * width)
        mask = self.softmax(mask).unsqueeze(3)
        #print(context.shape, mask.shape)
        context = torch.matmul(context, mask)
       #print(context.shape)
        context = context.view(batch, channel, 1, 1)
        context = self.conv2(context)
        context = self.ln(context)
        context = self.act(context)
        context = self.conv3(context)
        context = x * context.expand_as(x)

        return context

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None,
                 dilation = (1, 1), residual = True, BatchNorm = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding = dilation[0], dilation = dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes,
                             padding = dilation[1], dilation = dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 2
    """docstring for Bottleneck"""
    def __init__(self, inplanes, planes, stride = 1, downsample = None,
                 dilation = (1,1), residual = True, BatchNorm = None, expansion = 2):
        super(Bottleneck, self).__init__()
        self.conv1 =nn.Conv2d(inplanes, planes, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride = stride,
                               padding = dilation[1], bias = False, dilation = dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*expansion, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes*expansion)
        self.relu = Mish()
        self.downsample = downsample
        self.stride = stride
        self.gc = GClayer(planes*expansion)

    def forward(self, x):
        residual = x
        #print('x', x.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print('out',out.size())
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)


        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.gc(out)
        out += residual
        

        return out





class DRN(nn.Module):
    """docstring for DRN"""
    def __init__(self, block, layers, arch = 'D',
                 channels=(64,64,64,128,256,512,512,512),
                 BatchNorm = None):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_dim = channels[-1]
        self.arch = arch

        if arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(1, channels[0], 7, 1, 3, bias = False),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace = True))
            self.layer1 = self._make_conv_layers(channels[0], layers[0], 1, BatchNorm = BatchNorm)
            self.layer2 = self._make_conv_layers(channels[1], layers[1], 2, BatchNorm= BatchNorm)
            self.layer3 = self._make_layer(block, channels[2], layers[2], 2, BatchNorm = BatchNorm)
            self.layer4 = self._make_layer(block, channels[3], layers[3], 2, BatchNorm = BatchNorm)
            self.layer5 = self._make_layer(block, channels[4], layers[4], dilation = 2, new_level = False, BatchNorm = BatchNorm)
            self.layer6 = None if layers[5] == 0 else \
                self._make_layer(block, channels[5], layers[5], dilation = 4,
                                 new_level = False, BatchNorm = BatchNorm)
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation = 2, BatchNorm = BatchNorm)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation = 1, BatchNorm = BatchNorm)

            self._init_weight()

    def _make_conv_layers(self, channels, convs, stride = 1, dilation = 1, BatchNorm = None):
        modules = []
        padding = dilation
        for i in range(convs):
            if i == 0:
                modules.append(nn.Conv2d(self.inplanes, channels, 3, stride, padding, dilation, bias = False))
            else:
                modules.append(nn.Conv2d(self.inplanes, channels, 3, 1, padding, dilation, bias = False))

            modules.append(nn.BatchNorm2d(channels)) 
            modules.append(nn.ReLU(inplace = True))

            self.inplanes = channels
        return nn.Sequential(*modules)

    def _make_layer(self, block, planes, blocks, stride = 1, dilation = 1,
                new_level = True, residual = True, BatchNorm = None):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride !=1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes*block.expansion),
                )
        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation = (1,1) if dilation == 1 else(
                dilation // 2 if new_level else dilation, dilation),
            residual = residual, BatchNorm = BatchNorm))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual = residual,
                                dilation = (dilation, dilation), BatchNorm = BatchNorm))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.arch =='D':
            x = self.layer0(x)
        x = self.layer1(x)
        #low_level_feat = x
        #print('x1:', low_level_feat.size())
        x = self.layer2(x)
        
        x = self.layer3(x)
        low_level_feat = x
        #print('x2:', low_level_feat2.size())
        x = self.layer4(x)
        
        x = self.layer5(x)

        if self.layer6 is not None:
            x = self.layer6(x)
        if self.layer7 is not None:
            x = self.layer7(x)
        if self.layer8 is not None:
            x = self.layer8(x)
            #print(x.shape)


        return x


def gc_drn_54(BatchNorm, pretrained = False):
    model = DRN(Bottleneck, [1,1,3,4,6,3,1,1], arch = 'D', BatchNorm = BatchNorm)
        
    return model

if __name__ == "__main__":
    import torch
    model = drn_d_54(BatchNorm = nn.BatchNorm2d, pretrained = False)
    input = torch.rand(1,3, 128,256)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())