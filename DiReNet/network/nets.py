import numpy as np
import math

import torch 
import torch.nn as nn
from collections import OrderedDict

class netFC000(nn.Module):
    def __init__(self, reduction=4):
        super(netFC000, self).__init__()
        self.dim = 2048//2
        self.dimz = self.dim//reduction
        
        self.encoderX = nn.Sequential(
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dimz))
        self.encoderY = nn.Sequential(
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dimz))
        
        self.BX = nn.Sequential(
                            nn.Linear(self.dimz, 2*self.dimz),
                            nn.ReLU(),
                            nn.Linear(2*self.dimz, self.dimz))
        self.BY = nn.Sequential(
                            nn.Linear(self.dimz, 2*self.dimz),
                            nn.ReLU(),
                            nn.Linear(2*self.dimz, self.dimz))
        
        self.decoderX = nn.Sequential(
                            nn.Linear(self.dimz , self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim))       
        self.decoderY = nn.Sequential(
                            nn.Linear(self.dimz , self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim))

    def forward(self, data):  
        dataX = data[:,:, 0:16,:]
        dataY = data[:,:,16:32,:]
        n, c, h, w = dataX.detach().size()
        dataX = dataX.contiguous().view(n, -1) # 平坦化,reshape
        dataY = dataY.contiguous().view(n, -1)

        Zx = self.encoderX(dataX)
        Zy = self.encoderY(dataY)
        
        Zx = self.BX(Zx)
        Zy = self.BY(Zy)
        
        dataX_hat = self.decoderX(Zx)
        dataY_hat = self.decoderY(Zy)
        dataX_hat = dataX_hat.contiguous().view(n, c, h, w)
        dataY_hat = dataY_hat.contiguous().view(n, c, h, w)
        
        return Zx,Zy,dataX_hat,dataY_hat

class netFC00(nn.Module):
    def __init__(self, reduction=4):
        super(netFC00, self).__init__()
        self.dim = 2048//2
        self.dimz = self.dim//reduction
        
        self.encoderX = nn.Sequential(
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dimz))
        self.encoderY = nn.Sequential(
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dimz))
        
        self.decoderX = nn.Sequential(
                            nn.Linear(self.dimz , self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim))       
        self.decoderY = nn.Sequential(
                            nn.Linear(self.dimz , self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim))

    def forward(self, data):  
        dataX = data[:,:, 0:16,:]
        dataY = data[:,:,16:32,:]
        n, c, h, w = dataX.detach().size()
        dataX = dataX.contiguous().view(n, -1) # 平坦化,reshape
        dataY = dataY.contiguous().view(n, -1)

        Zx = self.encoderX(dataX)
        Zy = self.encoderY(dataY)
        
        dataX_hat = self.decoderX(Zx)
        dataY_hat = self.decoderY(Zy)
        dataX_hat = dataX_hat.contiguous().view(n, c, h, w)
        dataY_hat = dataY_hat.contiguous().view(n, c, h, w)
        
        return Zx,Zy,dataX_hat,dataY_hat

class netFC0(nn.Module):
    def __init__(self, reduction=4, rm=10 ,R=2):
        super(netFC0, self).__init__()
        self.dim = 2048//2
        self.dimz = self.dim//reduction
        self.dimout = self.dimz*10//rm
        self.xy_l = 2*self.dimout//R
        self.w_l = 2*self.dimout-2*self.xy_l
        
        self.encoderX = nn.Sequential(
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dimz))
        self.encoderY = nn.Sequential(
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dimz))
        
        self.MInn = nn.Sequential(
                            nn.Linear(2*self.dimz, 2*self.dimz),
                            nn.ReLU(),
                            nn.Linear(2*self.dimz, 2*self.dimout))
        
        self.decoderX = nn.Sequential(
                            nn.Linear(self.w_l+self.xy_l , self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim))       
        self.decoderY = nn.Sequential(
                            nn.Linear(self.w_l+self.xy_l , self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim))

    def forward(self, data):  
        dataX = data[:,:, 0:16,:]
        dataY = data[:,:,16:32,:]
        n, c, h, w = dataX.detach().size()
        dataX = dataX.contiguous().view(n, -1) # 平坦化,reshape
        dataY = dataY.contiguous().view(n, -1)

        Zx = self.encoderX(dataX)
        Zy = self.encoderY(dataY)
        
        Z = torch.cat((Zx,Zy), 1)
        Z = self.MInn(Z)
        
        Ww = Z[:, 0:self.w_l]
        Wx = Z[:, self.w_l : self.w_l+self.xy_l]
        Wy = Z[:, self.w_l+self.xy_l : 2*self.dimout]
        
        dataX_hat = self.decoderX(torch.cat((Ww,Wx), 1))
        dataY_hat = self.decoderY(torch.cat((Ww,Wy), 1))
        dataX_hat = dataX_hat.contiguous().view(n, c, h, w)
        dataY_hat = dataY_hat.contiguous().view(n, c, h, w)
        
        return Wx,Wy,dataX_hat,dataY_hat

class netFC1(nn.Module):
    def __init__(self, reduction=4):
        super(netFC1, self).__init__()
        self.dim = 2048
        self.encoder = nn.Sequential(
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim//reduction))
        self.decoder = nn.Sequential(
                            nn.Linear(self.dim//reduction, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim))

    def forward(self, data):   
        
        n, c, h, w = data.detach().size()
        data = data.view(n, -1) # 平坦化,reshape
        z = self.encoder(data)
        data_hat = self.decoder(z)
        data_hat = data_hat.view(n, c, h, w)
        
        return data_hat

class netFC2(nn.Module):
    def __init__(self, reduction=4):
        super(netFC2, self).__init__()
        self.dim = 2048
        self.encoder1 = nn.Sequential(
                            nn.Linear(self.dim//2, self.dim//2),
                            nn.ReLU(),
                            nn.Linear(self.dim//2, self.dim//2//reduction))
        self.encoder2 = nn.Sequential(
                            nn.Linear(self.dim//2, self.dim//2),
                            nn.ReLU(),
                            nn.Linear(self.dim//2, self.dim//2//reduction))
        self.decoder1 = nn.Sequential(
                            nn.Linear(self.dim//2//reduction, self.dim//2),
                            nn.ReLU(),
                            nn.Linear(self.dim//2, self.dim//2))
        self.decoder2 = nn.Sequential(
                            nn.Linear(self.dim//2//reduction, self.dim//2),
                            nn.ReLU(),
                            nn.Linear(self.dim//2, self.dim//2))

    def forward(self, data):   
        dataX = data[:,:, 0:16,:]
        dataY = data[:,:,16:32,:]
        n, c, h, w = dataX.detach().size()
        dataX = dataX.contiguous().view(n, -1) # 平坦化,reshape
        dataY = dataY.contiguous().view(n, -1)

        z1 = self.encoder1(dataX)
        dataX_hat = self.decoder1(z1)
        z2 = self.encoder2(dataY)
        dataY_hat = self.decoder2(z2)
    
        dataX_hat = dataX_hat.contiguous().view(n, c, h, w)
        dataY_hat = dataY_hat.contiguous().view(n, c, h, w)
        return dataX_hat, dataY_hat

class netFC3(nn.Module):
    def __init__(self, reduction=4, R=4):
        super(netFC3, self).__init__()
        self.dim = 2048
        self.dimout = self.dim//reduction
        self.xy_l = self.dimout//R
        self.w_l = self.dimout-2*self.xy_l
        
        self.encoder = nn.Sequential(
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dimout))
        
        self.decoderX = nn.Sequential(
                            nn.Linear(self.w_l+self.xy_l , self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim//2))
        
        self.decoderY = nn.Sequential(
                            nn.Linear(self.w_l+self.xy_l , self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim//2))

    def forward(self, data):  
        dataX = data[:,:, 0:16,:]
        dataY = data[:,:,16:32,:]
        n, c, h, w = dataX.detach().size()
        dataX = dataX.contiguous().view(n, -1) #reshape
        dataY = dataY.contiguous().view(n, -1)

        Z = self.encoder(torch.cat((dataX,dataY), 1))
        
        Zw = Z[:, 0:self.w_l]
        Zx = Z[:, self.w_l : self.w_l+self.xy_l]
        Zy = Z[:, self.w_l+self.xy_l : self.dimout]
        
        dataX_hat = self.decoderX(torch.cat((Zw,Zx), 1))
        dataY_hat = self.decoderY(torch.cat((Zw,Zy), 1))
        dataX_hat = dataX_hat.contiguous().view(n, c, h, w)
        dataY_hat = dataY_hat.contiguous().view(n, c, h, w)
        
        return dataX,dataY,Zx,Zy,dataX_hat,dataY_hat
