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

class IEBlock2(nn.Module):#short-cut
    def __init__(self, dimin=2, dimout=2):
        super(IEBlock2, self).__init__()
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
        self.identity = nn.Identity()
        self.conv1x1_2 = ConvGDN(dimin, dimout, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.conv1x1(out)
        identity = self.conv1x1_2(identity)
        out = self.relu(out+identity)
        return out
        
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

class vaenet2_E1D3(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E1D3, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock2(dimin=4, dimout=2)
        self.encoder_y = IEBlock2(dimin=4, dimout=2)
        self.encoder_w = IEBlock2(dimin=4, dimout=2)
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

class vaenet2_E2D3(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E2D3, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_w = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
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

class vaenet2_E3D3(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E3D3, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = nn.Sequential(OrderedDict([
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2)),
            ("MIE3", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_y = nn.Sequential(OrderedDict([
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2)),
            ("MIE3", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_w = nn.Sequential(OrderedDict([
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2)),
            ("MIE3", IEBlock2(dimin=2, dimout=2))
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

class vaenet2_E1D2(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E1D2, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock2(dimin=4, dimout=2)
        self.encoder_y = IEBlock2(dimin=4, dimout=2)
        self.encoder_w = IEBlock2(dimin=4, dimout=2)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2))
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

class vaenet2_E2D2(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E2D2, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_w = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
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
            ("MIBlock2", IDBlock(dim=2))
        ]))
        self.decoder_feature_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2))
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

class vaenet2_E1D3x2(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E1D3x2, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock2(dimin=4, dimout=2)
        self.encoder_y = IEBlock2(dimin=4, dimout=2)
        self.encoder_w = IEBlock2(dimin=4, dimout=2)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
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
        x_hat = self.decoder_feature_x1(x_hat)+self.decoder_feature_x2(x_hat)
        y_hat = self.decoder_feature_y1(y_hat)+self.decoder_feature_y2(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet2_E2D3x2(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E2D3x2, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_w = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
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
        x_hat = self.decoder_feature_x1(x_hat)+self.decoder_feature_x2(x_hat)
        y_hat = self.decoder_feature_y1(y_hat)+self.decoder_feature_y2(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet2_E1x2D3x2(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E1x2D3x2, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x1 = IEBlock2(dimin=4, dimout=2)
        self.encoder_x2 = IEBlock2(dimin=4, dimout=2)
        self.encoder_y1 = IEBlock2(dimin=4, dimout=2)
        self.encoder_y2 = IEBlock2(dimin=4, dimout=2)
        self.encoder_w1 = IEBlock2(dimin=4, dimout=2)
        self.encoder_w2 = IEBlock2(dimin=4, dimout=2)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
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

        xy = self.encoder_w1(torch.cat((data[:,:, 0:16,:], data[:,:,16:32,:]), dim=1))+self.encoder_w2(torch.cat((data[:,:, 0:16,:], data[:,:,16:32,:]), dim=1))
        x = self.encoder_x1(torch.cat((data[:,:, 0:16,:], xy), dim=1))+self.encoder_x2(torch.cat((data[:,:, 0:16,:], xy), dim=1))
        y = self.encoder_y1(torch.cat((data[:,:,16:32,:], xy), dim=1))+self.encoder_y2(torch.cat((data[:,:,16:32,:], xy), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h//2, w)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h//2, w)
        x_hat = self.decoder_feature_x1(x_hat)+self.decoder_feature_x2(x_hat)
        y_hat = self.decoder_feature_y1(y_hat)+self.decoder_feature_y2(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet2_E1D3x3(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E1D3x3, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock2(dimin=4, dimout=2)
        self.encoder_y = IEBlock2(dimin=4, dimout=2)
        self.encoder_w = IEBlock2(dimin=4, dimout=2)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y3 = nn.Sequential(OrderedDict([
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
        x_hat = self.decoder_feature_x1(x_hat)+self.decoder_feature_x2(x_hat)+self.decoder_feature_x3(x_hat)
        y_hat = self.decoder_feature_y1(y_hat)+self.decoder_feature_y2(y_hat)+self.decoder_feature_y3(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet2_E2D3x3(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E2D3x3, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_w = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y3 = nn.Sequential(OrderedDict([
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
        x_hat = self.decoder_feature_x1(x_hat)+self.decoder_feature_x2(x_hat)+self.decoder_feature_x3(x_hat)
        y_hat = self.decoder_feature_y1(y_hat)+self.decoder_feature_y2(y_hat)+self.decoder_feature_y3(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet2_E1D3x5(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E1D3x5, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock2(dimin=4, dimout=2)
        self.encoder_y = IEBlock2(dimin=4, dimout=2)
        self.encoder_w = IEBlock2(dimin=4, dimout=2)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x5 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y5 = nn.Sequential(OrderedDict([
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
        x_hat = self.decoder_feature_x1(x_hat)+self.decoder_feature_x2(x_hat)+self.decoder_feature_x3(x_hat)+self.decoder_feature_x4(x_hat)+self.decoder_feature_x5(x_hat)
        y_hat = self.decoder_feature_y1(y_hat)+self.decoder_feature_y2(y_hat)+self.decoder_feature_y3(y_hat)+self.decoder_feature_y4(y_hat)+self.decoder_feature_y5(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet2_E2D3x5(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E2D3x5, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_w = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x5 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y5 = nn.Sequential(OrderedDict([
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
        x_hat = self.decoder_feature_x1(x_hat)+self.decoder_feature_x2(x_hat)+self.decoder_feature_x3(x_hat)+self.decoder_feature_x4(x_hat)+self.decoder_feature_x5(x_hat)
        y_hat = self.decoder_feature_y1(y_hat)+self.decoder_feature_y2(y_hat)+self.decoder_feature_y3(y_hat)+self.decoder_feature_y4(y_hat)+self.decoder_feature_y5(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet2_E1D3x10(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E1D3x10, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock2(dimin=4, dimout=2)
        self.encoder_y = IEBlock2(dimin=4, dimout=2)
        self.encoder_w = IEBlock2(dimin=4, dimout=2)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x5 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x6 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x7 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x8 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x9 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x0 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y5 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y6 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y7 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y8 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y9 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y0 = nn.Sequential(OrderedDict([
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
        x_hat = self.decoder_feature_x1(x_hat)+self.decoder_feature_x2(x_hat)+self.decoder_feature_x3(x_hat)+self.decoder_feature_x4(x_hat)+self.decoder_feature_x5(x_hat)+self.decoder_feature_x6(x_hat)+self.decoder_feature_x7(x_hat)+self.decoder_feature_x8(x_hat)+self.decoder_feature_x9(x_hat)+self.decoder_feature_x0(x_hat)
        y_hat = self.decoder_feature_y1(y_hat)+self.decoder_feature_y2(y_hat)+self.decoder_feature_y3(y_hat)+self.decoder_feature_y4(y_hat)+self.decoder_feature_y5(y_hat)+self.decoder_feature_y6(y_hat)+self.decoder_feature_y7(y_hat)+self.decoder_feature_y8(y_hat)+self.decoder_feature_y9(y_hat)+self.decoder_feature_y0(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet2_E2D3x10(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet2_E2D3x10, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 12
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_w = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIE1", IEBlock2(dimin=4, dimout=2)),
            ("MIE2", IEBlock2(dimin=2, dimout=2))
        ]))
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x5 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x6 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x7 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x8 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x9 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x0 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y5 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y6 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y7 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y8 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y9 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y0 = nn.Sequential(OrderedDict([
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
        x_hat = self.decoder_feature_x1(x_hat)+self.decoder_feature_x2(x_hat)+self.decoder_feature_x3(x_hat)+self.decoder_feature_x4(x_hat)+self.decoder_feature_x5(x_hat)+self.decoder_feature_x6(x_hat)+self.decoder_feature_x7(x_hat)+self.decoder_feature_x8(x_hat)+self.decoder_feature_x9(x_hat)+self.decoder_feature_x0(x_hat)
        y_hat = self.decoder_feature_y1(y_hat)+self.decoder_feature_y2(y_hat)+self.decoder_feature_y3(y_hat)+self.decoder_feature_y4(y_hat)+self.decoder_feature_y5(y_hat)+self.decoder_feature_y6(y_hat)+self.decoder_feature_y7(y_hat)+self.decoder_feature_y8(y_hat)+self.decoder_feature_y9(y_hat)+self.decoder_feature_y0(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet32(nn.Module):
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

    def forward(self, data):
        n, c, h, w = data.detach().size()

        xy = self.encoder_w(torch.cat((data[:,:,:,0:16], data[:,:,:,16:32]), dim=1))
        x = self.encoder_x(torch.cat((data[:,:,:,0:16], xy), dim=1))
        y = self.encoder_y(torch.cat((data[:,:,:,16:32], xy), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h, w//2)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h, w//2)
        x_hat = self.decoder_feature_x(x_hat)
        y_hat = self.decoder_feature_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet32p(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet32p, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IEBlock2(dimin=4, dimout=2))
        ]))
        self.encoder_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IEBlock2(dimin=4, dimout=2))
        ]))
        self.encoder_w = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(4, 4, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IEBlock2(dimin=4, dimout=2))
        ]))
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_x = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_y = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))

    def forward(self, data):
        n, c, h, w = data.detach().size()

        xy = self.encoder_w(torch.cat((data[:,:,:,0:16], data[:,:,:,16:32]), dim=1))
        x = self.encoder_x(torch.cat((data[:,:,:,0:16], xy), dim=1))
        y = self.encoder_y(torch.cat((data[:,:,:,16:32], xy), dim=1))
                
        zx = self.encoder_fc_x(x.view(n, -1))
        zy = self.encoder_fc_y(y.view(n, -1))
        zw = self.encoder_fc_w(xy.view(n, -1))

        zxw = torch.cat((zx, zw), dim=1)
        zyw = torch.cat((zy, zw), dim=1)
        
        x_hat = self.decoder_fc_x(zxw).view(n, c, h, w//2)
        y_hat = self.decoder_fc_y(zyw).view(n, c, h, w//2)
        x_hat = self.decoder_x(x_hat)
        y_hat = self.decoder_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat

class vaenet32_E1D3x5(nn.Module):
    def __init__(self, reduction, R):
        super(vaenet32_E1D3x5, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        
        self.dimout = total_size//reduction
        self.z_l = self.dimout//R
        self.w_l = self.dimout-2*self.z_l
        
        self.encoder_x = IEBlock2(dimin=4, dimout=2)
        self.encoder_y = IEBlock2(dimin=4, dimout=2)
        self.encoder_w = IEBlock2(dimin=4, dimout=2)
        self.encoder_fc_x = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_y = nn.Linear(total_size//2, self.z_l)
        self.encoder_fc_w = nn.Linear(total_size//2, self.w_l)
        
        self.decoder_fc_x = nn.Linear(self.z_l+self.w_l, total_size//2)
        self.decoder_fc_y = nn.Linear(self.z_l+self.w_l, total_size//2)
        
        self.decoder_feature_x1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_x5 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y1 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y3 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y4 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvGDN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("MIBlock1", IDBlock(dim=2)),
            ("MIBlock2", IDBlock(dim=2)),
            ("MIBlock3", IDBlock(dim=2))
        ]))
        self.decoder_feature_y5 = nn.Sequential(OrderedDict([
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
        x_hat = self.decoder_feature_x1(x_hat)+self.decoder_feature_x2(x_hat)+self.decoder_feature_x3(x_hat)+self.decoder_feature_x4(x_hat)+self.decoder_feature_x5(x_hat)
        y_hat = self.decoder_feature_y1(y_hat)+self.decoder_feature_y2(y_hat)+self.decoder_feature_y3(y_hat)+self.decoder_feature_y4(y_hat)+self.decoder_feature_y5(y_hat)
        x_hat = self.sigmoid_x(x_hat)
        y_hat = self.sigmoid_y(y_hat)

        return xy,zw,zx,zy,x_hat,y_hat



def VAEnet(reduction=4):
    
    model = vaenet(reduction=reduction)
    return model
