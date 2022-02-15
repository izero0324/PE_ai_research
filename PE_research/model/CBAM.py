import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                #print(avg_pool.size())
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                #print(max_pool.size())
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F 
from model.Mish import Mish
#from Mish import Mish

def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output



class Bottleneck_base(nn.Module):
    expansion = 2
    """docstring for Bottleneck"""
    def __init__(self, inplanes, planes, stride = 1, downsample = None,
                 dilation = (1,1), residual = True, BatchNorm = None, expansion = 2, drop_connect_rate = 0.5):
        super(Bottleneck_base, self).__init__()
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
        self.drop_connect_rate = drop_connect_rate
        

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
        if self.drop_connect_rate:
            x = drop_connect(x, p = self.drop_connect_rate, training=self.training)
        out += residual
        #out = self.relu(out)

        return out


class DRN(nn.Module):
    """docstring for DRN"""
    def __init__(self, block, layers, arch = 'D',
                 channels=(16,32,64,128,256,512,512,512),
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

            self.at1 = CBAM(channels[0])
            self.at2 = CBAM(channels[1])
            self.at3 = CBAM(channels[2]*2)
            self.at4 = CBAM(channels[3]*2)
            self.at5 = CBAM(channels[4]*2)
            self.at6 = CBAM(channels[5]*2)
            self.at7 = CBAM(channels[6])
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
            drop_connect_rate = 0.25
            if drop_connect_rate:
                drop_connect_rate *= float(i) / blocks
            layers.append(block(self.inplanes, planes, residual = residual,
                                dilation = (dilation, dilation), BatchNorm = BatchNorm, drop_connect_rate = drop_connect_rate))

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
        x = self.at1(x)
        #low_level_feat = x
        #print('x1:', low_level_feat.size())
        x = self.layer2(x)
        x = self.at2(x)
        x = self.layer3(x)
        #print(x.shape)
        x = self.at3(x)
        #print('x2:', low_level_feat2.size())
        x = self.layer4(x)
        x = self.at4(x)
        
        x = self.layer5(x)
        x = self.at5(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            x = self.at6(x)
        if self.layer7 is not None:
            x = self.layer7(x)
            x = self.at7(x)
        if self.layer8 is not None:
            x = self.layer8(x)
            #print(x.shape)


        return x

class DRN_P(nn.Module):
    """docstring for DRN"""
    def __init__(self, block, layers, arch = 'D',
                 channels=(16,32,64,128,256,512,512,512),
                 BatchNorm = None):
        super(DRN_P, self).__init__()
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

            self.at1 = CBAM(channels[0])
            self.at2 = CBAM(channels[1])
            self.at3 = CBAM(channels[2]*2)
            self.at4 = CBAM(channels[3]*2)
            self.at5 = CBAM(channels[4]*2)
            self.at6 = CBAM(channels[5]*2)
            self.at7 = CBAM(channels[6])
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
            drop_connect_rate = 0.25
            if drop_connect_rate:
                drop_connect_rate *= float(i) / blocks
            layers.append(block(self.inplanes, planes, residual = residual,
                                dilation = (dilation, dilation), BatchNorm = BatchNorm, drop_connect_rate = drop_connect_rate))

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
        x = self.at1(x)
        #low_level_feat = x
        #print('x1:', low_level_feat.size())
        x = self.layer2(x)
        x = self.at2(x)
        x = self.layer3(x)
        #print(x.shape)
        #x = self.at3(x)
        #print('x2:', low_level_feat2.size())
        x = self.layer4(x)
        #x = self.at4(x)
        
        x = self.layer5(x)
        #x = self.at5(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            #x = self.at6(x)
        if self.layer7 is not None:
            x = self.layer7(x)
            x = self.at7(x)
        if self.layer8 is not None:
            x = self.layer8(x)
            #print(x.shape)


        return x


def drn_d_CBAM(BatchNorm, pretrained = False):
    model = DRN(Bottleneck_base, [1,1,3,4,6,3,1,1], arch = 'D', BatchNorm = BatchNorm)
        
    return model


def drn_d_CBAM_P(BatchNorm, pretrained = False):
    model = DRN_P(Bottleneck_base, [1,1,3,4,6,3,1,1], arch = 'D', BatchNorm = BatchNorm)
        
    return model

if __name__ == "__main__":
    import torch
    block = drn_d_CBAM(nn.BatchNorm2d)
    input = torch.rand(1,1, 512,512)
    output = block(input)
    print(output.size())