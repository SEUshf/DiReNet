import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from network.pytorch_gdn import GDN

class ConvGDN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        device = torch.device('cuda')
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvGDN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
#             ('bn', GDN(out_planes, device))
        ])) 

class IABlock(nn.Module):
    def __init__(self, dimin=2, dimout=2):
        super(IABlock, self).__init__()
        self.one = nn.Sequential(OrderedDict([
            ('conv3x3', ConvGDN(dimin, dimin, [3,3])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path1 = nn.Sequential(OrderedDict([
            ('conv1x3_1', ConvGDN(dimin, dimin, [1,3])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x1_2', ConvGDN(dimin, dimin, [3,1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5_1', ConvGDN(dimin, dimin, [1,5])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1_2', ConvGDN(dimin, dimin, [5,1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path3 = nn.Sequential(OrderedDict([
            ('conv1x7_1', ConvGDN(dimin, dimin, [1,7])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x1_2', ConvGDN(dimin, dimin, [7,1])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.conv1x1_out = nn.Sequential(OrderedDict([
            ('conv1x1', ConvGDN(dimin, dimout, 1)),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.conv1x1_in = nn.Sequential(OrderedDict([
            ('conv1x1', ConvGDN(dimin, dimin, 1)),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.identity = nn.Identity()

    def forward(self, x):
        x_id = self.identity(x)
        out0 = self.one(x)
        out1 = self.path1(out0)
        out2 = self.path2(out0)
        out3 = self.path3(out0)
        out_add = self.conv1x1_in(out0+out1+out2+out3)
        out_mut = out_add*x_id
        out = self.conv1x1_out(out_mut)
        return out
    
class IABlock2(nn.Module):
    def __init__(self, dim=2):
        super(IABlock2, self).__init__()
        self.nn = 2
        self.one = nn.Sequential(OrderedDict([
            ('conv3x3', ConvGDN(dim, self.nn*dim, [3,3])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path1 = nn.Sequential(OrderedDict([
            ('conv1x3_1', ConvGDN(self.nn*dim, self.nn*dim, [3,3])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x1_2', ConvGDN(self.nn*dim, self.nn*dim, [3,3])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5_1', ConvGDN(self.nn*dim, self.nn*dim, [5,5])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1_2', ConvGDN(self.nn*dim, self.nn*dim, [5,5])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path3 = nn.Sequential(OrderedDict([
            ('conv1x7_1', ConvGDN(self.nn*dim, self.nn*dim, [7,7])),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x1_2', ConvGDN(self.nn*dim, self.nn*dim, [7,7])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.conv1x1_out = nn.Sequential(OrderedDict([
            ('conv1x1', ConvGDN(dim, dim, 1)),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.conv1x1_in = nn.Sequential(OrderedDict([
            ('conv1x1', ConvGDN(self.nn*dim, dim, 1)),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.identity = nn.Identity()

    def forward(self, x):
        x_id = self.identity(x)
        out0 = self.one(x)
        out1 = self.path1(out0)
        out2 = self.path2(out0)
        out3 = self.path3(out0)
        out_add = self.conv1x1_in(out0+out1+out2+out3)
        out_mut = out_add*x_id
        out = self.conv1x1_out(out_mut)
        return out


class DiReNetE(nn.Module):
    def __init__(self):
        super(DiReNetE, self).__init__()

        self.encoder_x = IABlock(dimin=4, dimout=2)
        self.encoder_y = IABlock(dimin=4, dimout=2)
        self.encoder_w = IABlock(dimin=4, dimout=2)

    def forward(self, data):
        n, c, h, w = data.detach().size()

        xy = self.encoder_w(torch.cat((data[:,:, 0:16,:], data[:,:,16:32,:]), dim=1))
        x = self.encoder_x(torch.cat((data[:,:, 0:16,:], xy), dim=1))
        y = self.encoder_y(torch.cat((data[:,:,16:32,:], xy), dim=1))

        return xy,x,y

class DiReNetC(nn.Module):
    def __init__(self, reduction, R):
        super(DiReNetC, self).__init__()
        total_size = 2048
        
        self.dimall = total_size//reduction
        self.z_l = self.dimall//R
        self.w_l = self.dimall-2*self.z_l
        
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)

    def forward(self,xy,x,y):
        n, c, w, h= 200, 2, 32, 12

        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_c = self.decoder_fc_x(zxw).view(n, c, w//2, h)
        y_c = self.decoder_fc_y(zyw).view(n, c, w//2, h)

        return zw,zx,zy,x_c,y_c

class DiReNetC32(nn.Module):
    def __init__(self, reduction, R):
        super(DiReNetC32, self).__init__()
        total_size = 2048
        
        self.dimall = total_size//reduction
        self.z_l = self.dimall//R
        self.w_l = self.dimall-2*self.z_l
        
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)

    def forward(self,xy,x,y):
        n, c, w, h= 200, 2, 32, 32

        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_c = self.decoder_fc_x(zxw).view(n, c, w//2, h)
        y_c = self.decoder_fc_y(zyw).view(n, c, w//2, h)

        return zw,zx,zy,x_c,y_c

class DiReNetD3(nn.Module):
    def __init__(self):
        super(DiReNetD3, self).__init__()

        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_x3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_y3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, x_c, y_c):

        x_hat = self.decoder_feature_x1(x_c)+self.decoder_feature_x2(x_c)+self.decoder_feature_x3(x_c)
        y_hat = self.decoder_feature_y1(y_c)+self.decoder_feature_y2(y_c)+self.decoder_feature_y3(y_c)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return x_hat,y_hat

class DiReNetD5(nn.Module):
    def __init__(self):
        super(DiReNetD5, self).__init__()

        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_x3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_x4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_x5 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_y3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_y4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.decoder_feature_y5 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IABlock2(dim=2)),
            ("MIBlock2", IABlock2(dim=2)),
            ("MIBlock3", IABlock2(dim=2))
        ]))
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self,x_c,y_c):
 
        x_hat = self.decoder_feature_x1(x_c)+self.decoder_feature_x2(x_c)+self.decoder_feature_x3(x_c)+self.decoder_feature_x4(x_c)+self.decoder_feature_x5(x_c)
        y_hat = self.decoder_feature_y1(y_c)+self.decoder_feature_y2(y_c)+self.decoder_feature_y3(y_c)+self.decoder_feature_y4(y_c)+self.decoder_feature_y5(y_c)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return x_hat,y_hat

def DiReNet(reduction=4):
    
    model = DiReNetE()
    return model
