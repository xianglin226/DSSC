from platform import libc_ver
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch.conv import GraphConv, TAGConv, GATConv


# Domain-specific Batch Normalization
class DSBatchNorm(nn.Module):
    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        super(DSBatchNorm, self).__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domain)])
        
    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()
            
    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()
            
    def _check_input_dim(self, input):
        raise NotImplementedError
            
    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device) #, requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy()==i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]
#                 self.bns[i].training = False
#                 out[indices] = self.bns[i](x[indices])
#                 self.bns[i].training = True
        return out

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1, norm=None, dispersion="gene", dropout=0, dropoutFinal=0):
        super(Decoder, self).__init__()
        self.scale_layer = GATConv(in_feats=input_dim, out_feats=output_dim, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, activation=None, allow_zero_in_degree=True, bias=False)
        self.dispersion = dispersion
        if dispersion == "gene":
            self.px_r = nn.Parameter(torch.randn(output_dim), requires_grad=True)
        elif dispersion == "gene-batch":
            self.px_r = nn.Parameter(torch.randn(output_dim, norm), requires_grad=True)
        else:
            self.px_r = None

        self.mean_act = nn.Softmax(dim=-1)
        self.disp_act = DispAct()

        if dropoutFinal >0:
            self.dropout = nn.Dropout(dropoutFinal)
        else:
            self.dropout = None

    def forward(self, g, x, y=None):
        h_px_scale = torch.sum(self.scale_layer(g, x), dim=1)
        px_scale = self.mean_act(h_px_scale)
        if self.dropout:
            px_scale = self.dropout(px_scale)

        if self.dispersion == "gene":
            px_r = self.px_r
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(y, self.n_batch), self.px_r)
        else:
            px_r = None
        px_r = self.disp_act(self.px_r)
        return px_scale, px_r


class DecoderBN(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1, norm=None, dispersion="gene", dropout=0, dropoutFinal=0):
        super(DecoderBN, self).__init__()
        self.scale_layer = GATConv(in_feats=input_dim, out_feats=output_dim, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, activation=None, allow_zero_in_degree=True, bias=False)
        self.dispersion = dispersion
        if dispersion == "gene":
            self.px_r = nn.Parameter(torch.randn(output_dim), requires_grad=True)
        elif dispersion == "gene-batch":
            self.px_r = nn.Parameter(torch.randn(output_dim, norm), requires_grad=True)
        else:
            self.px_r = None

        self.n_batch = norm
        if type(norm) == int:
            if norm==1: # TO DO
                self.norm = nn.BatchNorm1d(output_dim)
            else:
                self.norm = DSBatchNorm(output_dim, norm)
        else:
            self.norm = None

#        self.mean_act = MeanAct()
        self.mean_act = nn.Softmax(dim=-1)
        self.disp_act = DispAct()

        if dropoutFinal >0:
            self.dropout = nn.Dropout(dropoutFinal)
        else:
            self.dropout = None

    def forward(self, g, x, y=None):
        h_px_scale = torch.sum(self.scale_layer(g, x), dim=1)
        if self.norm:
            if len(x) == 1:
                pass
            elif self.norm.__class__.__name__ == 'DSBatchNorm':
                h_px_scale = self.norm(h_px_scale, y)
            else:
                h_px_scale = self.norm(h_px_scale)
        px_scale = self.mean_act(h_px_scale)
        if self.dropout:
            px_scale = self.dropout(px_scale)

        if self.dispersion == "gene":
            px_r = self.px_r
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(y, self.n_batch), self.px_r)
        else:
            px_r = None
        px_r = self.disp_act(self.px_r)
        return px_scale, px_r


class ConditionalDecoderBN(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1, n_condition=1, norm=None, dispersion="gene", dropout=0):
        super(ConditionalDecoderBN, self).__init__()
        self.scale_layer = GATConv(in_feats=input_dim+n_condition-1, out_feats=output_dim, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, activation=None, allow_zero_in_degree=True, bias=False)
        self.dispersion = dispersion
        if dispersion == "gene":
            self.px_r = nn.Parameter(torch.randn(output_dim), requires_grad=True)
        elif dispersion == "gene-batch":
            self.px_r = nn.Parameter(torch.randn(output_dim, norm), requires_grad=True)
        else:
            self.px_r = None

        self.n_batch = norm
        if type(norm) == int:
            if norm==1: # TO DO
                self.norm = nn.BatchNorm1d(output_dim)
            else:
                self.norm = DSBatchNorm(output_dim, norm)
        else:
            self.norm = None

        self.mean_act = MeanAct()
        self.disp_act = DispAct()

        if dropout >0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, g, x, y=None):
        h_px_scale = torch.sum(self.scale_layer(g, x), dim=1)
        if self.norm:
            if len(x) == 1:
                pass
            elif self.norm.__class__.__name__ == 'DSBatchNorm':
                h_px_scale = self.norm(h_px_scale, y)
            else:
                h_px_scale = self.norm(h_px_scale)
        px_scale = self.mean_act(h_px_scale)
        if self.dropout:
            px_scale = self.dropout(px_scale)

        if self.dispersion == "gene":
            px_r = self.px_r
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(y, self.n_batch), self.px_r)
        else:
            px_r = None
        px_r = self.disp_act(self.px_r)
        return px_scale, px_r


class NBLoss(nn.Module):
    def __init__(self, reduction=True):
        super(NBLoss, self).__init__()
        self.eps = 1e-10
        self.reduction = reduction

    def forward(self, x, mean, disp, library_size=None):
        disp = disp.unsqueeze(0)
        library_size = library_size[:, None]
#        mean = mean / (torch.sum(mean, dim=1, keepdim=True)+self.eps) * library_size
        mean = mean * library_size

        t1 = torch.lgamma(x+1.0) + torch.lgamma(disp+self.eps) - torch.lgamma(x+disp+self.eps)
        t2 = (x+disp) * torch.log(1.0 + (mean/(disp+self.eps))) + (x * (torch.log(disp+self.eps) - torch.log(mean+self.eps)))
        nb_final = t1 + t2

        if self.reduction:
            nb_final = torch.mean(torch.sum(nb_final, dim=1))
        return nb_final


class GammaLoss(nn.Module):
    def __init__(self, reduction=True):
        super(GammaLoss, self).__init__()
        self.eps = 1e-10
        self.reduction = reduction

    def forward(self, x, mean, disp):
        t1 = x * disp / (mean+self.eps) - (disp-1) * torch.log(x+self.eps)
        t2 = disp * torch.log(mean / (disp+self.eps)) + torch.lgamma(disp+self.eps)
        gamma_final = t1 + t2

        if self.reduction:
            gamma_final = torch.mean(torch.sum(gamma_final, dim=1))
        return gamma_final


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.mean(torch.sum(1 + 2*logvar - mu**2 - logvar.exp().pow(2), dim=1))


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-6, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
'''
class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-4, max=1e4)
'''


class Bilinear(nn.Module):
    """
    The inner product decoder
    """
    def __init__(self, n_features, dropout, use_bias=False, activation=torch.sigmoid):
        super(Bilinear, self).__init__()
        self.dropout = dropout
        self.use_bias = use_bias
        self.activation = activation
        self.kernel = nn.Parameter(torch.zeros(size=(n_features, n_features), dtype= torch.float))
        nn.init.xavier_uniform_(self.kernel, gain=1.414)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(size=1, dtype= torch.float))

    def forward(self, input):
        x = input
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        h = torch.matmul(x, self.kernel)
        output = torch.matmul(h, x.T)

        if self.use_bias:
            output = output + self.bias

        return output

class InnerProduct(nn.Module):
    def __init__(self, dropout=0):
        super(InnerProduct, self).__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, input):
        x = self.drop(input)
        output = torch.matmul(x, x.T)
        return output

class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        
        result = torch.mean(result)
        return result

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x
