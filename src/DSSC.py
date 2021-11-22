from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import ZINBLoss, NBLoss, MeanAct, DispAct
import numpy as np
import math, os
from sklearn import metrics
from utils import cluster_acc
from scipy import stats, spatial
from utils import *

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if type=="encode" and i==len(layers)-1:
            break
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)
    
class Spatialmodel(nn.Module):
    def __init__(self, input_dim, z_dim, neg, encodeLayer=[], decodeLayer=[],
            activation="relu", sigma=1., alpha=1., gamma=1., ml_weight=1., fi=0.0001, cutoff=0.5):
        super(Spatialmodel, self).__init__()
        self.cutoff = cutoff
        self.activation = activation
        self.neg = neg
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.fi = fi
        self.ml_weight = ml_weight
        self.z_dim = z_dim
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self.encoder = buildNetwork([input_dim] + encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim]+decodeLayer, type="decode", activation=activation)
        #self.decoder = buildNetwork(decodeLayer, type="decode", activation=activation)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())
        self.NBLoss = NBLoss().cuda()
        self.zinb_loss = ZINBLoss().cuda()
        self.mse = nn.MSELoss()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
        
    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / self.alpha
        num = torch.pow(1.0 + num, -(self.alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p
     
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forward(self, x):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)
        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        num, lq = self.cal_latent(z0)
        return z0, q, num, lq, _mean, _disp, _pi
        
    def forward_AE(self, x):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)
        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        num, lq = self.cal_latent(z0)
        return z0, num, lq, _mean, _disp, _pi
        
    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs1 = Variable(xbatch)
            z,_,_,_,_,_ = self.forward_AE(inputs1)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return kldloss
    
    def kldloss(self, p, q):
        c1 = -torch.sum(p * torch.log(q))
        c2 = -torch.sum(p * torch.log(p))
        l = c1 - c2
        return l

    def pairwise_loss(self, p1, p2, cons_type):
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            return self.ml_weight*ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return self.cl_weight*cl_loss
            
    def pretrain_autoencoder(self, X, X_raw, X_sf, batch_size=256, lr=0.001, epochs=400, ae_save=False, ae_weights='AE_weights.pth.tar'):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(X_sf))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        count=0
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).cuda()
                x_raw_tensor = Variable(x_raw_batch).cuda()
                sf_tensor = Variable(sf_batch).cuda()
                _, z_num, lqbatch, mean_tensor, disp_tensor, pi_tensor = self.forward_AE(x_tensor) 
                #recon_loss = self.mse(mean_tensor, x_tensor)
                #recon_loss = self.NBLoss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, scale_factor=sf_tensor)
                recon_loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                lpbatch = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                lpbatch = lpbatch + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(lpbatch, lqbatch)
                if count > epochs *self.cutoff:
                      loss = recon_loss# + kl_loss * self.fi
                else:
                      loss = recon_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Pretrain epoch [{}/{}], MSE loss:{:.4f} kl loss:{:.4f}'.format(
                batch_idx+1, epoch+1, recon_loss.item(), kl_loss.item()))
            count+=1

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, X_sf, n_clusters, ml_ind1=np.array([]), ml_ind2=np.array([]),
            ml_p=1., y=None, lr=1., batch_size=256, num_epochs=1, update_interval=1, tol=1e-3, save_dir=""):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        print("Clustering stage")
        X = torch.tensor(X).float().cuda()
        X_sf = torch.tensor(X_sf).float().cuda()
        X_raw = torch.tensor(X_raw).cuda()
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim))
        #optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0001)
        
        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(n_clusters, n_init=20)
        Zdata = self.encodeBatch(X, batch_size=batch_size)
        self.y_pred = kmeans.fit_predict(Zdata.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
            print('Initializing KNN ACC= %.4f' % knn_ACC(self.neg, self.y_pred))
        
        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        ml_num_batch = int(math.ceil(1.0*ml_ind1.shape[0]/batch_size))
        ml_num = ml_ind1.shape[0]

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        update_ml = 1

        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                if y is not None:
                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print('Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch+1, acc, nmi, ari))
                    print('KNN ACC= %.4f' % knn_ACC(self.neg, self.y_pred))

                # save current model
                if (epoch>0 and delta_label < tol) or epoch%10 == 0:
                    self.save_checkpoint({'epoch': epoch+1,
                            'state_dict': self.state_dict(),
                            'mu': self.mu,
                            #'p': p,
                            #'q': q,
                            'y_pred': self.y_pred,
                            'y_pred_last': self.y_pred_last,
                            #'y': y
                            }, epoch+1, filename=save_dir)

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch>0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break


            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            kl_loss_val = 0.0
            for batch_idx in range(num_batch):
                x_batch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw_batch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_sf_batch = X_sf[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(x_batch)
                sfinputs = Variable(x_sf_batch)
                rawinputs = Variable(x_raw_batch)
                target = Variable(pbatch)

                #zbatch, h1_, qbatch, z_num, lqbatch = self.forward(inputs1)
                zbatch, qbatch, z_num, lqbatch, mean_tensor, disp_tensor, pi_tensor = self.forward(inputs)
                cluster_loss = self.cluster_loss(target, qbatch)
                #recon_loss = self.mse(mean_tensor, inputs)
                #recon_loss = self.NBLoss(x=rawinputs, mean=mean_tensor, disp=disp_tensor, scale_factor=sfinputs)
                recon_loss = self.zinb_loss(x=rawinputs, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sfinputs)
                target2 = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                target2 = target2 + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(target2, lqbatch) * self.fi
                loss = recon_loss + cluster_loss * self.gamma #+ kl_loss  * self.fi
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs)
                recon_loss_val += recon_loss.data * len(inputs)
                kl_loss_val += kl_loss.data * len(inputs)
                #y_loss_val += y_loss.data * len(inputs) 
                train_loss = recon_loss_val + cluster_loss_val + kl_loss_val #+ y_loss_val

            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f MSE Loss: %.4f kl Loss: %.4f" % (
               epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num, kl_loss_val / num))
            
            ml_loss = 0.0
            if epoch % update_ml == 0:
                for ml_batch_idx in range(ml_num_batch):
                    px1 = X[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    pxraw1 = X_raw[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    sf1 = X_sf[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    px2 = X[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    sf2 = X_sf[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    pxraw2 = X_raw[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    rawinputs1 = Variable(pxraw1)
                    sfinput1 = Variable(sf1)
                    inputs2 = Variable(px2)
                    rawinputs2 = Variable(pxraw2)
                    sfinput2 = Variable(sf2)
                    z1, q1, _, _, mean1, disp1, pi1 = self.forward(inputs1)
                    z2, q2, _, _, mean2, disp2, pi2 = self.forward(inputs2)
                    loss = (ml_p*self.pairwise_loss(q1, q2, "ML")+self.zinb_loss(rawinputs1, mean1, disp1, pi1, sfinput1) + self.zinb_loss(rawinputs2, mean2, disp2, pi2, sfinput2))
                    #loss = (ml_p*self.pairwise_loss(q1, q2, "ML")+self.NBLoss(rawinputs1, mean1, disp1, sfinput1) + self.NBLoss(rawinputs2, mean2, disp2, sfinput2))
                    #loss = (ml_p*self.pairwise_loss(q1, q2, "ML")+self.mse(mean1, inputs1) + self.mse(mean2, inputs2))
                    # 0.1 for mnist/reuters, 1 for fashion, the parameters are tuned via grid search on validation set
                    ml_loss += loss.data
                    loss.backward()
                    optimizer.step()

            if ml_num_batch >0:
                print("ML loss", float(ml_loss.cpu()))

        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch
