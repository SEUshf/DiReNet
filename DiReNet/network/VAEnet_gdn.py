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
            # ('bn', nn.BatchNorm2d(out_planes))
            ('bn', GDN(out_planes, device))
        ])) 

        
class IEBlock(nn.Module):
    def __init__(self, dimin=2, dimout=2):
        super(IEBlock, self).__init__()
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
        self.conv1x1 = ConvGDN(3*dimin, dimout, 1)
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
            ('conv3x3_1', ConvGDN(dim, 4*dim, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x3_2', ConvGDN(4*dim, 4*dim, 3)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv5x5_1', ConvGDN(dim, 2*dim, 5)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x5_2', ConvGDN(2*dim, 2*dim, 5)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.path3 = nn.Sequential(OrderedDict([
            ('conv7x7_1', ConvGDN(dim, 2*dim, 7)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x7_2', ConvGDN(2*dim, 2*dim, 7)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.conv1x1 = ConvGDN(8*dim, dim, 1)
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


class vaenet(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock(dimin=4, dimout=2)
        self.encoder_y = IEBlock(dimin=4, dimout=2)
        self.encoder_w = IEBlock(dimin=4, dimout=2)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
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

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet_plus(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet_plus, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock3", IEBlock(dimin=4, dimout=2))
        ]))
        self.encoder_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock3", IEBlock(dimin=4, dimout=2))
        ]))
        self.encoder_w = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock3", IEBlock(dimin=4, dimout=2))
        ]))
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
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

        return xy,zw,zx,zy,x_hat,y_hat


class vaenet_xw(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet_xw, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock(dimin=4, dimout=2)
#         self.encoder_y = IEBlock(dimin=4, dimout=2)
        self.encoder_w = IEBlock(dimin=4, dimout=2)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
#         self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
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
        y = self.encoder_x(torch.cat((data[:,:,16:32,:], xy), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_x(y.view(n, -1))
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
    
class vaenet32(nn.Module):## WynerVAE   w zx zy 维度一样 # wyner MI
    def __init__(self, reduction, R):
        super(vaenet32, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock(dimin=4, dimout=2)
        self.encoder_y = IEBlock(dimin=4, dimout=2)
        self.encoder_w = IEBlock(dimin=4, dimout=2)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
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

        return xy,zw,zx,zy,x_hat,y_hat

def VAEnet(reduction=4):
    
    model = vaenet(reduction=reduction)
    return model
