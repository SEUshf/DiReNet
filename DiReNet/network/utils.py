import numpy as np
import math
import torch 
import torch.nn as nn
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
            ('bn', nn.BatchNorm2d(out_planes)), 
            ('relu',nn.LeakyReLU(negative_slope=0.3, inplace=True))
        ]))

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.direct_path = nn.Sequential(OrderedDict([
            ("conv_1", ConvBN(2, 8, kernel_size=3)),
            ("conv_2", ConvBN(8, 16, kernel_size=3)),
            ("conv_3", nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)),
            ("bn", nn.BatchNorm2d(2))
        ]))
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
    def forward(self, x):
        identity = self.identity(x)
        out = self.direct_path(x)
        out = self.relu(out + identity)
        return out

class CsiNet(nn.Module):
    def __init__(self,reduction=4):
        super(CsiNet, self).__init__()
        total_size, in_channel, h, w = 2048, 2, 32, 12
        dim_out = total_size // reduction

        self.encoder_convbn = ConvBN(in_channel, 2, kernel_size=3)
        self.encoder_fc = nn.Linear(total_size, dim_out)
        
        self.decoder_fc = nn.Linear(dim_out, total_size)
        self.decoder_RefineNet1 = RefineNet()
        self.decoder_RefineNet2 = RefineNet()
        self.decoder_conv = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)    

    def forward(self, x):
        n, c, h, w = x.detach().size()
        x = self.encoder_convbn(x)
        x = x.view(n,-1) # 
        x = self.encoder_fc(x)       
        
        x = self.decoder_fc(x)
        x = x.view(n, c, h, w)
        x = self.decoder_RefineNet1(x)
        x = self.decoder_RefineNet2(x)
        x = self.decoder_conv(x)

        return x

class two_fc(nn.Module):
    def __init__(self, reduction=4):
        super(two_fc, self).__init__()
        self.dim = 2048
        self.encoder = nn.Sequential(
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim//reduction))
        self.decoder = nn.Sequential(
                            nn.Linear(self.dim//reduction, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim))

    def forward(self, data):   
        
        n, c, h, w = data.detach().size()
        data = data.view(n, -1) 
        z = self.encoder(data)
        data_hat = self.decoder(z)
        data_hat = data_hat.view(n, c, h, w)
        
        return data_hat
    
class two_fc_XY(nn.Module):
    def __init__(self, reduction=4, R=4):
        super(two_fc_XY, self).__init__()
        self.dim = 2048
        self.dimout = self.dim//reduction
        self.xy_l = self.dimout//R
        self.w_l = self.dimout-2*self.xy_l
        
        self.encoder = nn.Sequential(
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
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
        dataX = dataX.contiguous().view(n, -1) 
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

class two_fc_XY_baseline(nn.Module):
    def __init__(self, reduction=4):
        super(two_fc_XY_baseline, self).__init__()
        self.dim = 2048
        self.dimout = self.dim//reduction
        
        self.encoder = nn.Sequential(
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dimout))
        
        self.decoderX = nn.Sequential(
                            nn.Linear(self.dimout//2 , self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim//2))
        
        self.decoderY = nn.Sequential(
                            nn.Linear(self.dimout//2 , self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim),
                            nn.ReLU(),
                            nn.Linear(self.dim, self.dim//2))

    def forward(self, data):  
        dataX = data[:,:, 0:16,:]
        dataY = data[:,:,16:32,:]
        n, c, h, w = dataX.detach().size()
        dataX = dataX.contiguous().view(n, -1) 
        dataY = dataY.contiguous().view(n, -1)

        Z = self.encoder(torch.cat((dataX,dataY), 1))
        
        Zx = Z[:, 0:self.dimout//2]
        Zy = Z[:, self.dimout//2 : self.dimout]
        
        dataX_hat = self.decoderX(Zx)
        dataY_hat = self.decoderY(Zy)
        dataX_hat = dataX_hat.contiguous().view(n, c, h, w)
        dataY_hat = dataY_hat.contiguous().view(n, c, h, w)
        
        return dataX_hat,dataY_hat

    
class GaussianSampler(nn.Module):
    def __init__(self, dim, para_list = None):
        super(GaussianSampler, self).__init__()
        self.dim = dim
        if para_list is None:
            para_list = [0.55] * dim
        self.p_theta_ = torch.nn.Parameter(torch.tensor(para_list, requires_grad = True))
        
    def get_trans_mat(self):
        p_theta = self.p_theta_.cuda().unsqueeze(-1)
        #p_theta = torch.softmax(p_theta, dim = 0)

        trans_row1 = torch.cat((torch.sin(p_theta),torch.cos(p_theta)), dim=-1).unsqueeze(-1)
        trans_row2 = torch.cat((torch.cos(p_theta),torch.sin(p_theta)), dim=-1).unsqueeze(-1)  #[dim, 2,1]
        return torch.cat((trans_row1, trans_row2), dim=-1)  #[dim,2,2]

    def gen_samples(self, num_sample, cuda = True):
        noise= torch.randn(self.dim,num_sample,2).cuda()
        trans_mat = self.get_trans_mat()
        samples = torch.bmm(noise, trans_mat).transpose(0,1) #[dim, nsample, 2]
        if not cuda:
            samples = samples.cpu().detach().numpy()
        return samples[:,:,0], samples[:,:,1] 

    def get_covariance(self):
        p_theta = self.p_theta_.cuda()
        return (2.*torch.sin(p_theta)*torch.cos(p_theta))

    def get_MI(self):
        rho = self.get_covariance()
        return -1./2.*torch.log(1-rho**2).sum().item()
    
class CLUBSample(nn.Module): 
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

    
class NWJ(nn.Module):   
    def __init__(self, x_dim, y_dim, hidden_size):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
                                    
    def forward(self, x_samples, y_samples): 
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))-1.  #shape [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1) - np.log(sample_size)).exp().mean() 
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


    
class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)