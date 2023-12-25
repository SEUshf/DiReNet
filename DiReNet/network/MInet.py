import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out
    
class IEBlock2(nn.Module):
    def __init__(self, dim=2):
        super(IEBlock2, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv1x3_1', ConvBN(dim, dim, [1,3])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x1_2', ConvBN(dim, dim, [3,1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5_1', ConvBN(dim, dim, [1,5])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1_2', ConvBN(dim, dim, [5,1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path3 = nn.Sequential(OrderedDict([
            ('conv1x7_1', ConvBN(dim, dim, [1,7])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x1_2', ConvBN(dim, dim, [7,1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.conv1x1 = ConvBN(3*dim, 2, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.conv1x1(out)
        out = self.relu(out)
        return out
    
class IEBlock(nn.Module):
    def __init__(self, dim=2):
        super(IEBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv1x3_1', ConvBN(dim, dim, [1,3])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x1_2', ConvBN(dim, dim, [3,1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5_1', ConvBN(dim, dim, [1,5])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1_2', ConvBN(dim, dim, [5,1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path3 = nn.Sequential(OrderedDict([
            ('conv1x7_1', ConvBN(dim, dim, [1,7])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x1_2', ConvBN(dim, dim, [7,1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.conv1x1 = ConvBN(3*dim, dim, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.conv1x1(out)
        out = self.relu(out)
        return out
    
class IDBlock(nn.Module):
    def __init__(self, dim=2):
        super(IDBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3_1', ConvBN(dim, 4*dim, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x3_2', ConvBN(4*dim, 4*dim, 3)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv5x5_1', ConvBN(dim, 2*dim, 5)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x5_2', ConvBN(2*dim, 2*dim, 5)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path3 = nn.Sequential(OrderedDict([
            ('conv7x7_1', ConvBN(dim, 2*dim, 7)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x7_2', ConvBN(2*dim, 2*dim, 7)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.conv1x1 = ConvBN(8*dim, dim, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.conv1x1(out)
        out = self.relu(out + identity)
        return out
    
class MIBlock(nn.Module):
    def __init__(self, dim=2):
        super(MIBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3_1', ConvBN(dim, 4*dim, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x3_2', ConvBN(4*dim, 4*dim, 3)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x3_3', ConvBN(4*dim, 4*dim, 3)),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv7x7_1', ConvBN(dim, 4*dim, 7)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x7_2', ConvBN(4*dim, 4*dim, 7)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x7_3', ConvBN(4*dim, 4*dim, 7)),
        ]))
        self.conv1x1 = nn.Sequential(OrderedDict([
            ('conv1', ConvBN(8*dim, 4*dim, 5)),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv2', ConvBN(4*dim, dim, 1)),
        ]))
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out

class MIBlock2(nn.Module):
    def __init__(self, dim=2):
        super(MIBlock2, self).__init__()
        self.conv7x7 = ConvBN(dim, dim, 7)
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3_1', ConvBN(dim, 4*dim, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x3_2', ConvBN(4*dim, 4*dim, 3)),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv5x5_1', ConvBN(dim, 4*dim, 5)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x5_2', ConvBN(4*dim, 4*dim, 5)),
        ]))
        self.conv1x1 = ConvBN(8*dim, dim, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)
        out = self.conv7x7(x)
        out1 = self.path1(out)
        out2 = self.path2(out)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out

class MIBlock3(nn.Module):
    def __init__(self, dim=2):
        super(MIBlock3, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3_1', ConvBN(dim, 4*dim, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1_2', ConvBN(4*dim, 4*dim, [5,1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x5_3', ConvBN(4*dim, 4*dim, [1,5])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv7x7_1', ConvBN(dim, 4*dim, 7)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1_2', ConvBN(4*dim, 4*dim, [9,1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9_3', ConvBN(4*dim, 4*dim, [1,9])),
        ]))
        self.conv1x1 = ConvBN(8*dim, dim, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out

class MIBlocks(nn.Module):
    def __init__(self, dim=2):
        super(MIBlocks, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3_1', ConvBN(dim, dim, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x3_2', ConvBN(dim, dim, 3)),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv7x7_1', ConvBN(dim, dim, 7)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x7_2', ConvBN(dim, dim, 7)),
        ]))
        self.conv1x1 = nn.Sequential(OrderedDict([
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv2', ConvBN(2*dim, dim, 1)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

    def forward(self, x):
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.conv1x1(out)
        
        return out
    
class MINet1(nn.Module):#CRNet改   De*2   CRBlock+1
    def __init__(self, reduction=4, R=5):
        super(MINet1, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.encoder_fc_x = nn.Linear(total_size, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)

        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock()),
            ("CRBlock3", CRBlock())
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock()),
            ("CRBlock3", CRBlock())
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        n, c, h, w = data.detach().size()

        encode1 = self.encoder1(data)
        encode2 = self.encoder2(data)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        
        zx = self.encoder_fc_x(out.view(n, -1))
        zy = self.encoder_fc_y(out.view(n, -1))
        zw = self.encoder_fc_w(out.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        ZX = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        ZY = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(ZX)
        y_hat = self.decoder_feature_y(ZY)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat

class MINet12(nn.Module):#CRNet改   De*2   CRBlock+1  -de x==y
    def __init__(self, reduction=4, R=5):
        super(MINet12, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.encoder_fc_x = nn.Linear(total_size, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)

        self.decoder_fc = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock()),
            ("CRBlock3", CRBlock())
        ]))

        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        encode1 = self.encoder1(data)
        encode2 = self.encoder2(data)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        
        zx = self.encoder_fc_x(out.view(n, -1))
        zy = self.encoder_fc_y(out.view(n, -1))
        zw = self.encoder_fc_w(out.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        ZX = self.decoder_fc(zxw).view(n, c, h//2, w)
        ZY = self.decoder_fc(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature(ZX)
        y_hat = self.decoder_feature(ZY)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat
    
class MINet2(nn.Module):#CRBlock改成MIBlock
    def __init__(self, reduction=4, R=5):
        super(MINet2, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.encoder_fc_x = nn.Linear(total_size, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)

        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", MIBlock(dim=2)),
            ("MIBlock2", MIBlock(dim=2)),
            ("MIBlock3", MIBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", MIBlock(dim=2)),
            ("MIBlock2", MIBlock(dim=2)),
            ("MIBlock3", MIBlock(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        n, c, h, w = data.detach().size()

        encode1 = self.encoder1(data)
        encode2 = self.encoder2(data)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        
        zx = self.encoder_fc_x(out.view(n, -1))
        zy = self.encoder_fc_y(out.view(n, -1))
        zw = self.encoder_fc_w(out.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        ZX = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        ZY = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(ZX)
        y_hat = self.decoder_feature_y(ZY)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat

class MINet22(nn.Module):#CRBlock改成MIBlock  -去掉初始化
    def __init__(self, reduction=4, R=5):
        super(MINet22, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.encoder_fc_x = nn.Linear(total_size, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)

        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", MIBlock(dim=2)),
            ("MIBlock2", MIBlock(dim=2)),
            ("MIBlock3", MIBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", MIBlock(dim=2)),
            ("MIBlock2", MIBlock(dim=2)),
            ("MIBlock3", MIBlock(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        encode1 = self.encoder1(data)
        encode2 = self.encoder2(data)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        
        zx = self.encoder_fc_x(out.view(n, -1))
        zy = self.encoder_fc_y(out.view(n, -1))
        zw = self.encoder_fc_w(out.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        ZX = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        ZY = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(ZX)
        y_hat = self.decoder_feature_y(ZY)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat

class MINet222(nn.Module):#CRBlock改成MIBlock  -去掉初始化 -MIBlock改成3*MIBlock2
    def __init__(self, reduction=4, R=5):
        super(MINet222, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.encoder_fc_x = nn.Linear(total_size, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)

        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", MIBlock2(dim=2)),
            ("MIBlock2", MIBlock2(dim=2)),
            ("MIBlock3", MIBlock2(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", MIBlock2(dim=2)),
            ("MIBlock2", MIBlock2(dim=2)),
            ("MIBlock3", MIBlock2(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        encode1 = self.encoder1(data)
        encode2 = self.encoder2(data)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        
        zx = self.encoder_fc_x(out.view(n, -1))
        zy = self.encoder_fc_y(out.view(n, -1))
        zw = self.encoder_fc_w(out.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        ZX = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        ZY = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(ZX)
        y_hat = self.decoder_feature_y(ZY)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat

class MINet223(nn.Module):#CRBlock改成MIBlock  -去掉初始化 -MIBlock改成3*MIBlock3
    def __init__(self, reduction=4, R=5):
        super(MINet223, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.encoder_fc_x = nn.Linear(total_size, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)

        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("MIBlock1", MIBlock3(dim=2)),
            ("MIBlock2", MIBlock3(dim=2)),
            ("MIBlock3", MIBlock3(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("MIBlock1", MIBlock3(dim=2)),
            ("MIBlock2", MIBlock3(dim=2)),
            ("MIBlock3", MIBlock3(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        encode1 = self.encoder1(data)
        encode2 = self.encoder2(data)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        
        zx = self.encoder_fc_x(out.view(n, -1))
        zy = self.encoder_fc_y(out.view(n, -1))
        zw = self.encoder_fc_w(out.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        ZX = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        ZY = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(ZX)
        y_hat = self.decoder_feature_y(ZY)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat

class MINet224(nn.Module):#decoder x == decoder y
    def __init__(self, reduction=4, R=5):
        super(MINet224, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.encoder_fc_x = nn.Linear(total_size, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)

        self.decoder_fc = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", MIBlock(dim=2)),
            ("MIBlock2", MIBlock(dim=2)),
            ("MIBlock3", MIBlock(dim=2))
        ]))
        
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        encode1 = self.encoder1(data)
        encode2 = self.encoder2(data)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        
        zx = self.encoder_fc_x(out.view(n, -1))
        zy = self.encoder_fc_y(out.view(n, -1))
        zw = self.encoder_fc_w(out.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        ZX = self.decoder_fc(zxw).view(n, c, h//2, w)
        ZY = self.decoder_fc(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature(ZX)
        y_hat = self.decoder_feature(ZY)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat
    
class MINet3(nn.Module):#net22 改encoder ==> 3 parts
    def __init__(self, reduction=4, R=5):
        super(MINet3, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = MIBlocks(dim=2)
        self.encoder_y = MIBlocks(dim=2)
        self.encoder_w = MIBlocks(dim=4)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", MIBlock(dim=2)),
            ("MIBlock2", MIBlock(dim=2)),
            ("MIBlock3", MIBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", MIBlock(dim=2)),
            ("MIBlock2", MIBlock(dim=2)),
            ("MIBlock3", MIBlock(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        x = self.encoder_x(data[:,:, 0:16,:])
        y = self.encoder_y(data[:,:,16:32,:])
        xy = self.encoder_w(torch.cat((x, y), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(x_hat)
        y_hat = self.decoder_feature_y(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat

class MINet33(nn.Module):#net22 改encoder ==> 3 parts -en x == en y  -de x == de y
    def __init__(self, reduction=4, R=5):
        super(MINet33, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder = MIBlocks(dim=2)
        self.encoder_w = MIBlocks(dim=4)
        
        self.encoder_fc = nn.Linear(total_size//2, self.z_l)

        self.encoder_fc_w = nn.Linear(total_size, self.w_l)
        
        self.decoder_fc = nn.Linear(self.z_l+self.w_l, total_size//2)
    
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", MIBlock(dim=2)),
            ("MIBlock2", MIBlock(dim=2)),
            ("MIBlock3", MIBlock(dim=2))
        ]))

        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        x = self.encoder(data[:,:, 0:16,:])
        y = self.encoder(data[:,:,16:32,:])
        xy = self.encoder_w(torch.cat((x, y), dim=1))
                
        zx = self.encoder_fc(x.view(n, -1))
        zy = self.encoder_fc(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature(x_hat)
        y_hat = self.decoder_feature(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat

class MINet4(nn.Module):#en--de #3 parts
    def __init__(self, reduction=4, R=5):
        super(MINet4, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock(dim=2)
        self.encoder_y = IEBlock(dim=2)
        self.encoder_w = IEBlock(dim=4)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        x = self.encoder_x(data[:,:, 0:16,:])
        y = self.encoder_y(data[:,:,16:32,:])
        xy = self.encoder_w(torch.cat((x, y), dim=1))
#         xy = self.encoder_w(torch.cat((data[:,:, 0:16,:], data[:,:,16:32,:]), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(x_hat)
        y_hat = self.decoder_feature_y(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat

class MINet40(nn.Module):#data --> w
    def __init__(self, reduction=4, R=5):
        super(MINet40, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock(dim=2)
        self.encoder_y = IEBlock(dim=2)
        self.encoder_w = IEBlock(dim=4)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        x = self.encoder_x(data[:,:, 0:16,:])
        y = self.encoder_y(data[:,:,16:32,:])
        xy = self.encoder_w(torch.cat((data[:,:, 0:16,:], data[:,:,16:32,:]), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(x_hat)
        y_hat = self.decoder_feature_y(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat
    
class MINet42(nn.Module):#en--de #3 parts
    def __init__(self, reduction=4, R=5):
        super(MINet42, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder = IEBlock(dim=2)
        self.encoder_w = IEBlock(dim=4)
        self.encoder_fc = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)
        
        self.decoder_fc = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))

        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        x = self.encoder(data[:,:, 0:16,:])
        y = self.encoder(data[:,:,16:32,:])
        xy = self.encoder_w(torch.cat((x, y), dim=1))
                
        zx = self.encoder_fc(x.view(n, -1))
        zy = self.encoder_fc(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature(x_hat)
        y_hat = self.decoder_feature(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat

class MINet43(nn.Module):#en--de #3 parts
    def __init__(self, reduction=4, R=5):
        super(MINet43, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder = IEBlock(dim=2)
        self.encoder_w = IEBlock(dim=4)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))

        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        x = self.encoder(data[:,:, 0:16,:])
        y = self.encoder(data[:,:,16:32,:])
        xy = self.encoder_w(torch.cat((x, y), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature(x_hat)
        y_hat = self.decoder_feature(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat
    
class MINet5(nn.Module):## WynerVAE   w zx zy 维度一样
    def __init__(self, reduction=4, R=5):
        super(MINet5, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock2(dim=4)
        self.encoder_y = IEBlock2(dim=4)
        self.encoder_w = IEBlock2(dim=4)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        xy = self.encoder_w(torch.cat((data[:,:, 0:16,:], data[:,:, 0:16,:]), dim=1))
        x = self.encoder_x(torch.cat((data[:,:, 0:16,:], xy), dim=1))
        y = self.encoder_y(torch.cat((data[:,:, 0:16,:], xy), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(x_hat)
        y_hat = self.decoder_feature_y(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return zx,zy,x_hat,y_hat


class MINet51(nn.Module):## WynerVAE   w zx zy 维度一样 # wyner MI
    def __init__(self, reduction=4, R=5):
        super(MINet51, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock2(dim=4)
        self.encoder_y = IEBlock2(dim=4)
        self.encoder_w = IEBlock2(dim=4)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        xy = self.encoder_w(torch.cat((data[:,:, 0:16,:], data[:,:,16:32,:]), dim=1))
        x = self.encoder_x(torch.cat((data[:,:, 0:16,:], xy), dim=1))
        y = self.encoder_y(torch.cat((data[:,:,16:32,:], xy), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(x_hat)
        y_hat = self.decoder_feature_y(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zx,zy,x_hat,y_hat

class MINet52(nn.Module):## 2048
    def __init__(self, reduction=4, R=4):
        super(MINet52, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock2(dim=4)
        self.encoder_y = IEBlock2(dim=4)
        self.encoder_w = IEBlock2(dim=4)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        xy = self.encoder_w(torch.cat((data[:,:, 0:16,:], data[:,:,16:32,:]), dim=1))
        x = self.encoder_x(torch.cat((data[:,:, 0:16,:], xy), dim=1))
        y = self.encoder_y(torch.cat((data[:,:,16:32,:], xy), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(x_hat)
        y_hat = self.decoder_feature_y(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zx,zy,x_hat,y_hat
    
class MINet55(nn.Module):## WynerVAE：w zx zy维度一样  ##wyner MI   ### size增大
    def __init__(self, reduction=4, R=5):
        super(MINet55, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(4, 4, 3)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("IE1", IEBlock2(dim=4)),
            ("IE2", IEBlock(dim=2))
        ]))
        self.encoder_y = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(4, 4, 3)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("IE1", IEBlock2(dim=4)),
            ("IE2", IEBlock(dim=2))
        ]))
        self.encoder_w = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(4, 4, 3)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("IE1", IEBlock2(dim=4)),
            ("IE2", IEBlock(dim=2))
        ]))
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        xy = self.encoder_w(torch.cat((data[:,:, 0:16,:], data[:,:,16:32,:]), dim=1))
        x = self.encoder_x(torch.cat((data[:,:, 0:16,:], xy), dim=1))
        y = self.encoder_y(torch.cat((data[:,:,16:32,:], xy), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(x_hat)
        y_hat = self.decoder_feature_y(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zx,zy,x_hat,y_hat

class MINet0(nn.Module):#final
    def __init__(self, reduction=4, R=4):
        super(MINet0, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock(dim=2)
        self.encoder_y = IEBlock(dim=2)
        self.encoder_w = IEBlock(dim=4)

        self.encoder_xx = IEBlock2(dim=6)
        self.encoder_yy = IEBlock2(dim=6)
        
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, data):
        n, c, h, w = data.detach().size()

        x = self.encoder_x(data[:,:, 0:16,:])
        y = self.encoder_y(data[:,:,16:32,:])
        xy = self.encoder_w(torch.cat((x, y), dim=1))
        
        x = self.encoder_xx(torch.cat((x, xy), dim=1))
        y = self.encoder_yy(torch.cat((y, xy), dim=1))
        
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x(x_hat)
        y_hat = self.decoder_feature_y(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

def minet(reduction=4):
    
    model = MINet1(reduction=reduction)
    return model
