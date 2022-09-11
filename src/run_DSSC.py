import math, os
from time import time
from DSSC import DSSC
import numpy as np
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from scipy import sparse
import h5py
import pandas as pd
import scanpy as sc
import torch
from preprocess import concat_data, preprocessing_rna, norm_adj, SNN_adj, pearson_residuals
from utils import *
from scipy import stats, spatial
import scipy.sparse as ssp

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='sample_151507.h5', help = 'input data file')
     parser.add_argument('--FS_file', default="realdata/sample_151507_featureSelection_Index2000.csv", help='Spatial-based feature selection file')
    parser.add_argument('--select_genes', default=1000, type=int, help = 'number of HVGs used in clustering')
    parser.add_argument('--knn', default=20, type=int, help = 'K value for building KNN graph')
    parser.add_argument('--train_iter', default=400, type=int, help = 'iterations of pretraining')
    parser.add_argument('--n_clusters', default=-1, type=int, help = 'number of clusters')
    parser.add_argument('--lr', default=0.001, type=float, help = 'learning rate')
    parser.add_argument('--dropoutE', default=0., type=float, help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0., type=float, help='dropout probability for decoder')
    parser.add_argument('--encodeLayer', nargs="+", default=[128], type=int, help = 'encoder layer size')
    parser.add_argument('--decodeLayer', nargs="+", default=[128], type=int, help = 'decoder layer size')
    parser.add_argument('--encodeHead', default=3, type=int, help = 'number of encoder heads')
    parser.add_argument('--concat', default=True, type=bool, help = 'concatencate or avergae multiple heads')
    parser.add_argument('--z_dim', default=32, type=int, help = 'dimension of latent space')
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--save_dir', default='model_save')
    parser.add_argument('--final_latent_file', default=-1, help = 'output embedding layer or not')
    parser.add_argument('--final_labels', default=-1, help = 'output predicted label or not')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_ml', default=1, type=int, help='number of must-link loss used in each training'))
    parser.add_argument('--weight_ml', default = 0., type = float)
    parser.add_argument('--n_cl', default=1, type=int, help='number of cannot-link loss used in each training')
    parser.add_argument('--weight_cl', default = 0., type = float)
    parser.add_argument('--gamma', default=0.01, type=float, help='coefficient of clustering loss')
    parser.add_argument('--clustering_iters', default=200, type=int, help='iteration of clustering stage')
    parser.add_argument('--act', default="Adam", help='activation function')
    parser.add_argument('--sigma', default=0.1, type=float, help='noise added on data for denoising autoencoder')
    parser.add_argument('--ml_file', default=-1, help='the file of must-link')
    parser.add_argument('--cl_file', default=-1, help='the file of cannot-link')
    parser.add_argument('--run', default=1)

    args = parser.parse_args()
    print(args)
    ###read data
    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    pos = np.array(data_mat['Pos'])
    pos = pos.T
    y = np.array(data_mat['Y']) #if availble
    data_mat.close()
    
    ###remove NA cells
    f = np.where(y.astype(np.str) != "NA")[0]
    y = y[f]
    x = x[f,:]
    pos = pos[f,:]
    pos = pos.astype(np.float)
    
    ###Cluster number defined by user or calculated from y (if availble)
    if args.n_clusters == -1:
        n_clusters = np.shape(np.unique(y))[0]
    else:
        n_clusters = args.n_clusters
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    ###read gene filter file 
    filter = np.loadtxt(args.FS_file) # read featureselection file here
    filter = filter.astype(np.int)   
    x = x[:,filter]

    ###preprocessing scRNA-seq read counts matrix
    print("***Data prepocessing***")
    obs = pd.DataFrame()
    obs['cell_labels'] = pd.Categorical(y, categories=np.unique(y), ordered=False)
    obs['batch'] = pd.Categorical(np.repeat(1, x.shape[0]), categories=[1], ordered=False)
    obs['pos_x'] = pos[:,0]
    obs['pos_y'] = pos[:,1]
    adata = sc.AnnData(X=x, obs=obs)
    adata, count_X = preprocessing_rna(adata=adata, n_top_features=args.select_genes)
    pos_ = np.zeros([adata.n_obs, 2])
    pos_[:,0] = adata.obs["pos_x"]
    pos_[:,1] = adata.obs["pos_y"]
    y = np.array(adata.obs["cell_labels"].cat.codes)
    print("Size of after filtering", adata.X.shape)
    print("Size of after filtering", count_X.shape)
    
    #############################################################################################
    print("***Building graph***")
    ###knn
    A = kneighbors_graph(pos_, args.knn, mode="connectivity", metric="euclidean", include_self=True, n_jobs=-1)
    A = A.toarray()    
    A = sparse.csr_matrix(A)
    A_n = norm_adj(A)
    
    ###Get kneighbors_graph from spatial dist (this is just for evaluation)
    k0 = 20 #we now use a k=20 graph to evaluate
    A0 = kneighbors_graph(pos_, k0, mode="connectivity", metric="euclidean", include_self=False, n_jobs=-1)
    A0 = A0.toarray()
    
    dist = spatial.distance_matrix(pos_,pos_)
    p_ = []
    for i in range(dist.shape[0]):
        idx = np.argpartition(dist[i], k0)[:k0]
        p_.append(idx)

    ###############################################################################################################
    print("***Building constraints***")
    ###read ml 
    print("Read ML file")
    anchors_ml = np.loadtxt(args.ml_file)
    anchors_ml = anchors_ml.astype(np.int)
    #index -1 since here we build constraints from R
    ml_ind1 = anchors_ml[:,0] - 1
    ml_ind2 = anchors_ml[:,1] - 1 
    
    ###read cl
    print("Read CL file")
    anchors_cl = np.loadtxt(args.cl_file)
    anchors_cl = anchors_cl.astype(np.int)
    #index -1 since here we build constraints from R
    cl_ind1 = anchors_cl[:,0] - 1
    cl_ind2 = anchors_cl[:,1] - 1
    
    ###build model
    model = stGAE(input_dim=adata.n_vars, encodeLayer=args.encodeLayer, decodeLayer=args.decodeLayer, encodeHead=args.encodeHead, 
            encodeConcat=args.concat, gamma=args.gamma, activation="elu", z_dim=args.z_dim, sigma = args.sigma,
            dropoutE=args.dropoutE, dropoutD=args.dropoutD, device=args.device).to(args.device)

    print(str(model))

    t0 = time()
    
    ###pretraining stage   
    model.train_model(adata.X, A_n, A, count_X, adata.obs.size_factors, 
                    lr=args.lr, train_iter=args.train_iter, verbose=True, save_dir=args.save_dir)
    print('Pret-raining time: %d seconds.' % int(time() - t0))
    
    ###clustering stage
    y_pred, final_loss, epoch = model.fit(X=adata.X, X_raw = count_X, X_sf=adata.obs.size_factors, A = A, A_n = A_n,
            n_clusters = n_clusters, num_epochs=args.clustering_iters, y=y, n_ml = args.n_ml, n_cl = args.n_cl,
            ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2, ml_p=args.weight_ml, cl_p=args.weight_cl,
            p_=p_, lr = args.lr, update_interval=1, tol=0.001, save_dir=args.save_dir)

    t1 = time()
    print("Time used is:" + str(t1-t0))
    
    ###evaluation if y is available
    acc, nmi, ari = eval_cluster(y, y_pred)
    Iscore = Iscore_label(y_pred+1., A0)
    ka = knn_ACC(p_, y_pred)
    print('Final Clustering: ACC= %.4f, NMI= %.4f, ARI= %.4f, kNN_ACC= %.4f, I_score= %.4f, Loss = %.8f, Epoch = %.i' % (acc, nmi, ari, ka, Iscore, final_loss, epoch))
    
    ###output predicted labels and embedding
    if args.final_latent_file != -1:
         final_latent = model.encodeBatch(torch.tensor(adata.X, dtype=torch.float32), A_n).data.cpu().numpy()
         np.savetxt(args.save_dir + "/" + args.final_latent_file + "_" + str(args.run), final_latent, delimiter=",")   
    if args.final_labels != -1:
         np.savetxt(args.save_dir + "/" + args.final_labels + "_" + str(args.run), y_pred, delimiter=",")
    
