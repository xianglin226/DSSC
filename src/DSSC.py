import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from layers import DecoderBN, ConditionalDecoderBN, NBLoss, KLLoss, InnerProduct, ZINBLoss, MeanAct, DispAct
import numpy as np
import math, os
import dgl
from dgl.nn.pytorch.conv import GraphConv, TAGConv, GATConv
from utils import *
from sklearn.cluster import KMeans

eps = 1e-10
MAX_LOGVAR = 15

def buildNetwork(layers, activation="relu", dropout=0.):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        elif activation=="lrelu":
            net.append(nn.LeakyReLU(negative_slope=0.2))
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
    return nn.Sequential(*net)

class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_heads=1, dropout=0, concat=False, residual=False):
        super(GATEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats=input_dim, out_feats=hidden_dims[0], num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, residual=residual, activation=F.elu, allow_zero_in_degree=True))
        if concat:
            self.layers.append(nn.BatchNorm1d(hidden_dims[0]*num_heads))
            self.layers.append(nn.ELU())
            for i in range(1, len(hidden_dims)):
                self.layers.append(GATConv(in_feats=hidden_dims[i-1]*num_heads, out_feats=hidden_dims[i], num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, residual=residual, activation=F.elu, allow_zero_in_degree=True))
                self.layers.append(nn.BatchNorm1d(hidden_dims[i]*num_heads))
                self.layers.append(nn.ELU())
            self.enc_mu = GATConv(in_feats=hidden_dims[-1]*num_heads, out_feats=output_dim, num_heads=num_heads, feat_drop=0, attn_drop=0, residual=residual, activation=None, allow_zero_in_degree=True)
        else:
            self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
            self.layers.append(nn.ELU())
            for i in range(1, len(hidden_dims)):
                self.layers.append(GATConv(in_feats=hidden_dims[i-1], out_feats=hidden_dims[i], num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, residual=residual, activation=F.elu, allow_zero_in_degree=True))
                self.layers.append(nn.BatchNorm1d(hidden_dims[i]))
                self.layers.append(nn.ELU())
            self.enc_mu = GATConv(in_feats=hidden_dims[-1], out_feats=output_dim, num_heads=num_heads, feat_drop=0, attn_drop=0, residual=residual, activation=None, allow_zero_in_degree=True)
        self.dropout = dropout
        self.concat = concat
        self.hidden_dims = hidden_dims

    def forward(self, g, x):
        if self.concat:
            for i in range(0, len(self.hidden_dims)):
                x = self.layers[3*i](g, x)
                x = x.view(x.shape[0], x.shape[1]*x.shape[2])
                x = self.layers[3*i+1](x)
                x = self.layers[3*i+2](x)
        else:
            for i in range(0, len(self.hidden_dims)):
                x = self.layers[3*i](g, x)
                x = torch.sum(x, dim=1)
                x = self.layers[3*i+1](x)
                x = self.layers[3*i+1](x)
        mean = torch.sum(self.enc_mu(g, x), dim=1)
        return mean

class DSSC(nn.Module):
    def __init__(self, input_dim, encodeLayer=[], decodeLayer=[], encodeHead=3, encodeConcat=False, 
            activation='elu', z_dim=32, alpha=1., gamma=0.1, sigma=0.1, device="cuda"):
        super(DSSC, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        self.input_dim = input_dim
        self.encodeLayer = encodeLayer
        self.decodeLayer = decodeLayer
        self.encodeConcat = encodeConcat
        self.z_dim = z_dim
        self.num_encoderLayer = len(encodeLayer)+1
        self.activation = activation
        self.encoder = GATEncoder(input_dim=input_dim, hidden_dims=encodeLayer, output_dim=z_dim, num_heads=encodeHead, dropout=0.)
        self.decoder = buildNetwork([z_dim]+decodeLayer, activation=activation, dropout=0.)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())
        self.nb_loss = NBLoss().to(device)
        self.zinb_loss = ZINBLoss().to(device)
        self.device = device

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def aeForward(self, g, x):
        x_c = x + torch.randn_like(x) * self.sigma
        z = self.encoder(g, x_c)
        x_ = self.decoder(z)
        mean = self._dec_mean(x_)
        disp = self._dec_disp(x_)
        pi = self._dec_pi(x_)
        ##
        z0 = self.encoder(g, x)
        return z0, mean, disp, pi

    def aeForward2(self, g, x):
        x_c = x + torch.randn_like(x) * self.sigma
        z = self.encoder(g, x_c)
        x_ = self.decoder(z)
        mean = self._dec_mean(x_)
        disp = self._dec_disp(x_)
        pi = self._dec_pi(x_)    
        ##
        z0 = self.encoder(g, x)
        q = self.soft_assign(z0)
        return z0, q, mean, disp, pi

    def encodeForward(self, g, x):
        mu = self.encoder(g, x)
        return mu

    def encodeBatch(self, X, A):
        self.eval()
        G = dgl.from_scipy(A)
        G.ndata['feat'] = X
        X_v = Variable(X).to(self.device)
        G_v = G.to(self.device)
        mu = self.encodeForward(G_v, X_v)
        return mu.data

    def train_model(self, X, A_n, A, X_raw, size_factor, lr=0.001, train_iter=400, verbose=True):
        X = torch.tensor(X, dtype=torch.float32)
        G = dgl.from_scipy(A_n)
        G.ndata['feat'] = X
        X_raw = torch.tensor(X_raw, dtype=torch.float32)
        size_factor = torch.tensor(size_factor, dtype=torch.float32)

        X_v = Variable(X).to(self.device)
        G_v = G.to(self.device)
        A_v = Variable(torch.tensor(A.toarray(), dtype=torch.float32)).to(self.device)
        X_raw_v = Variable(X_raw).to(self.device)
        size_factor_v = Variable(size_factor).to(self.device)
        num = X.shape[0]
        optim_adam = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        print("Training")
        for i in range(train_iter):
            self.train()
            _, mean_tensor, disp_tensor, pi_tensor = self.aeForward(G_v, X_v)
            loss_zinb = self.zinb_loss(x=X_raw_v, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=size_factor_v)
            loss = loss_zinb
            self.zero_grad()
            loss.backward()
            optim_adam.step()

            if verbose:
                self.eval()
                loss_zinb_val = 0
                _, mean_tensor, disp_tensor, pi_tensor = self.aeForward(G_v, X_v)
                loss_zinb = self.zinb_loss(x=X_raw_v, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=size_factor_v)
                loss_zinb_val = loss_zinb.data
                print('Iteration:{}, ZINB loss:{:.8f}'.format(i+1, loss_zinb_val/num))

    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
        
    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return kldloss

    def pairwise_loss(self, p1, p2, cons_type):
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            return ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return cl_loss
     
    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def fit(self, X, X_raw, X_sf, A, A_n, n_clusters,
            p_, n_ml=0, n_cl=0, ml_ind1=np.array([]), ml_ind2=np.array([]), cl_ind1=np.array([]), cl_ind2=np.array([]),
            ml_p=1., cl_p=1., y=None, lr=0.001, batch_size=256, num_epochs=1, update_interval=1, tol=1e-3):
        '''X: tensor data'''
        
        X = torch.tensor(X, dtype=torch.float32)
        G = dgl.from_scipy(A_n)
        G.ndata['feat'] = X
        X_raw = torch.tensor(X_raw, dtype=torch.float32)
        X_sf = torch.tensor(X_sf, dtype=torch.float32)

        X_v = Variable(X).to(self.device)
        G_v = G.to(self.device)
        A_v = Variable(torch.tensor(A.toarray(), dtype=torch.float32)).to(self.device)
        X_raw_v = Variable(X_raw).to(self.device)
        X_sf_v = Variable(X_sf).to(self.device)
        
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))
              
        print("Initializing cluster centers with kmeans")
        kmeans = KMeans(n_clusters, n_init=20)
        Zdata = self.encodeBatch(X, A_n).data.cpu().numpy()
        self.y_pred = kmeans.fit_predict(Zdata)
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        
        if y is not None:   
            acc, nmi, ari = eval_cluster(y, self.y_pred)
            print('Initializing kmeans: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
            print('Initializing kmeans KNN ACC= %.4f' % knn_ACC(p_, self.y_pred))
       
        #build ml
        ml_pool1 = ml_ind1 
        ml_pool2 = ml_ind2 
        cl_pool1 = cl_ind1 
        cl_pool2 = cl_ind2 
        inds = np.random.choice(ml_pool1.shape[0], n_ml, replace=False)
        ml_ind1 = ml_pool1[inds]
        ml_ind2 = ml_pool2[inds]
        print("Constraints summary: ML=%.0f" % (len(ml_ind1)))
        #build cl
        inds = np.random.choice(cl_pool1.shape[0], n_cl, replace=False)
        cl_ind1 = cl_pool1[inds]
        cl_ind2 = cl_pool2[inds]
        print("Constraints summary: CL=%.0f" % (len(cl_ind1)))
        
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        ml_num_batch = int(math.ceil(1.0*ml_ind1.shape[0]/batch_size))
        ml_num = ml_ind1.shape[0]
        cl_num_batch = int(math.ceil(1.0*cl_ind1.shape[0]/batch_size))
        cl_num = cl_ind1.shape[0]

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        update_ml = 1
        update_cl = 1
        
        optim_adam = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        for epoch in range(num_epochs):
            self.eval()
            if epoch%update_interval == 0:
                # update the targe distribution p
                Zdata = self.encodeBatch(X, A_n).to(self.device)
                q = self.soft_assign(Zdata)
                p = self.target_distribution(q).data
                # evalute the clustering performance
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                                       
            if y is not None:
                acc, nmi, ari = eval_cluster(y, self.y_pred)
                final_acc, final_nmi, final_ari = acc, nmi, ari
                print('DEC Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch+1, acc, nmi, ari))
                        
            delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
            self.y_pred_last = self.y_pred
            if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break
                    
            #update links
            if epoch%update_ml == 0:
            #update ml            
               inds = np.random.choice(ml_pool1.shape[0], n_ml, replace=False)
               ml_ind1 = ml_pool1[inds]
               ml_ind2 = ml_pool2[inds]   
            if epoch%update_cl == 0:
            #update cl
               inds = np.random.choice(cl_pool1.shape[0], n_cl, replace=False)
               cl_ind1 = cl_pool1[inds]
               cl_ind2 = cl_pool2[inds]
                    
            #update clustering
            self.train()
            target = Variable(p).to(self.device)
            _, qbatch, mean_tensor, disp_tensor, pi_tensor = self.aeForward2(G_v, X_v)
            loss_cluster = self.cluster_loss(target, qbatch)
            loss_zinb = self.zinb_loss(x=X_raw_v, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=X_sf_v)
            loss = loss_zinb + self.gamma * loss_cluster
            optim_adam.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mu, 1)
            optim_adam.step()
            total_loss = loss_zinb.data/num + loss_cluster.data/num            
                      
            #update links
            if epoch%2==0:
              _, qbatch, mean_tensor, disp_tensor, pi_tensor = self.aeForward2(G_v, X_v)
              loss_zinb = self.zinb_loss(x=X_raw_v, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=X_sf_v)
              ml_q1 = qbatch[ml_ind1]
              ml_q2 = qbatch[ml_ind2]
              ml_loss = self.pairwise_loss(ml_q1, ml_q2, "ML")
              loss = ml_p * ml_loss + loss_zinb
              optim_adam.zero_grad()
              loss.backward()
              torch.nn.utils.clip_grad_norm_(self.mu, 1)
              optim_adam.step()
              total_loss = loss_zinb.data/num + loss_cluster.data/num + ml_loss.data/ml_num
              print('Clustering Iteration:{}, Total loss:{:.8f}, ZINB loss:{:.8f}, Cluster loss:{:.8f}, ML loss:{:.8f}'.format(epoch+1, 
                      total_loss, loss_zinb.data/num, loss_cluster.data/num, ml_loss.data/ml_num))

            else:
              _, qbatch, mean_tensor, disp_tensor, pi_tensor = self.aeForward2(G_v, X_v)
              loss_zinb = self.zinb_loss(x=X_raw_v, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=X_sf_v)
              cl_q1 = qbatch[cl_ind1]
              cl_q2 = qbatch[cl_ind2]
              cl_loss = self.pairwise_loss(cl_q1, cl_q2, "CL")
              loss = cl_p * cl_loss + loss_zinb
              optim_adam.zero_grad()
              loss.backward()
              torch.nn.utils.clip_grad_norm_(self.mu, 1)
              optim_adam.step()
              total_loss = loss_zinb.data/num + loss_cluster.data/num + cl_loss.data/cl_num
              print('Clustering Iteration:{}, Total loss:{:.8f}, ZINB loss:{:.8f}, Cluster loss:{:.8f}, CL loss:{:.8f}'.format(epoch+1, 
                      total_loss, loss_zinb.data/num, loss_cluster.data/num, cl_loss.data/cl_num))

        return self.y_pred, total_loss, epoch+1
