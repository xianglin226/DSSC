from time import time
import math, os
from utils import *
import scanpy as sc
from sklearn import metrics
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import collections
import h5py
from preprocess import read_dataset, normalize
from scipy import stats, spatial
from scDCC_CYCIF import cycifmodel
import sys


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='train',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='./osmFISH_SScortex_mouse.H5')
    parser.add_argument('--n_pairwise', default=100000, type=int)
    parser.add_argument('--weight_ml', default = 0.1, type = float)
    parser.add_argument('--dir_name', default = 'results_cycif')
    parser.add_argument('--n_clusters', default=11, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--fi', default=0., type=float, help='coefficient of KL loss')
    parser.add_argument('--sigma', default=2.5, type=float)
    parser.add_argument('--ae_weight_file', default='AE_weights_1.pth.tar')
    parser.add_argument('--embedding_file', default=-1, type=int)
    parser.add_argument('--prediction_file', default=-1, type=int)
    parser.add_argument('--neb', default=20, type=int)
    parser.add_argument('--filter', default=-1, type=int)
    parser.add_argument('--n_features', default=1000, type=int)
    parser.add_argument('--saveFeatures', default=-1, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    
    args = parser.parse_args()
    
    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    DIST_markers = np.array(data_mat["Loc"])
    DIST_markers = np.transpose(DIST_markers)
    data_mat.close()
    print(x.shape)
    print(DIST_markers.shape)
    neb = int(args.neb)
    
    A = kneighbors_graph(DIST_markers, 30, mode="connectivity", metric="euclidean", include_self=False, n_jobs=-1)
    A = A.toarray()
    if args.filter != -1:
         adata0 = sc.AnnData(x)
         adata0 = read_dataset(adata0,transpose=False,test_split=False,copy=True) 
         adata0 = normalize(adata0,size_factors=True,normalize_input=True,logtrans_input=True)
         score = []
         for i in range(adata0.X.shape[1]):
             gene = adata0.X[:,i]
             I = Iscore_gene(gene, A)
             score.append(I)
             if i%100==0:
                print(i)            
         score_ = np.argpartition(score, -args.n_features)[-args.n_features:]
         x = adata0.raw.X[:,score_]
    
    print(x.shape)
    
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
             transpose=False,
             test_split=False,
             copy=True) 

    adata = normalize(adata,
             size_factors=True,
             normalize_input=True,
             logtrans_input=True)

    input_size = adata.n_vars
    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)
    
    #spatial dist
    dist = spatial.distance_matrix(DIST_markers,DIST_markers)
    p_ = []
    for i in range(dist.shape[0]):
        idx = np.argpartition(dist[i], neb)[:neb]
        p_.append(idx)

    #Constraints
    n_pairwise = args.n_pairwise
    ml_ind1_1, ml_ind2_1 = generate_random_pair_from_neighbor2(p_, n_pairwise, neb)

    ml_ind1 = ml_ind1_1
    ml_ind2 = ml_ind2_1
    
    #Build model
    model = cycifmodel(input_dim=input_size, z_dim=32, neg=p_,
            encodeLayer=[256,64], decodeLayer=[64,256], sigma=args.sigma, gamma=args.gamma,
            ml_weight=args.weight_ml).cuda()

    model.pretrain_autoencoder(X=adata.X, X_raw = adata.raw.X, X_sf=adata.obs.size_factors, batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights="RU_ALL_weights.pth.tar")

    if not os.path.exists(args.dir_name):
            os.makedirs(args.dir_name)

    #get k
    latent = model.encodeBatch(torch.tensor(adata.X).cuda(), batch_size=args.batch_size).cpu().numpy()
    
    if args.n_clusters == -1:
       n_clusters = GetCluster(latent, res=1., n=30)
    else:
       print("n_cluster is defined as " + str(args.n_clusters))
       n_clusters = args.n_clusters
    
    kmeans = KMeans(n_clusters, n_init=20)
    y_p = kmeans.fit_predict(adata.X)
    acc = np.round(cluster_acc(y, y_p), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_p), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_p), 5)
    Iscore = Iscore_label(y_p+1., A)
    ka = knn_ACC(p_, y_p)
    print('Raw Kmeans Clustering： ACC= %.4f, NMI= %.4f, ARI= %.4f, kNN_ACC= %.4f, I_score= %.4f' % (acc, nmi, ari, ka, Iscore))
    
    kmeans = KMeans(n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(latent)
    np.savetxt(args.dir_name+"/Initial_y_pred.txt", y_pred, delimiter="\t")
 
    y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw = adata.raw.X, X_sf=adata.obs.size_factors, n_clusters = n_clusters, 
            batch_size=args.batch_size, num_epochs=1000, y=y,
            ml_ind1=ml_ind1, ml_ind2=ml_ind2, lr = 1.,
            update_interval=1, tol=0.001, save_dir=args.dir_name)
    
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    Iscore = Iscore_label(y_pred+1., A)
    ka = knn_ACC(p_, y_pred)
    print('Final Clustering： ACC= %.4f, NMI= %.4f, ARI= %.4f, kNN_ACC= %.4f, I_score= %.4f' % (acc, nmi, ari, ka, Iscore))
    
    file = args.data_file.split("/")[2]
    file = file.split(".")[0]
    
    final_latent = model.encodeBatch(torch.tensor(adata.X).cuda(), batch_size=args.batch_size).cpu().numpy()
    if args.embedding_file != -1:
         np.savetxt(args.dir_name+"/" + file + "." + "FINAL_latent.csv", final_latent, delimiter=",")
    if args.prediction_file != -1:
         np.savetxt(args.dir_name+"/" + file + "." + "y_pred.txt", y_pred, delimiter="\t")
    if args.saveFeatures != -1:
         np.savetxt(args.dir_name+"/" + file + "." + "featureSelection.txt", score_, delimiter="\t")
		    
