import scanpy as sc
import h5py
import numpy as np
from utils import *
from preprocess0 import read_dataset, normalize
from sklearn.neighbors import kneighbors_graph
from joblib import Parallel, delayed

#x is the count matrix; 
#Loc is the spatial coordinates
#Here we use the sample 151507 as an example, y is just used to remove unknown cells
sample = '151507'
inputfile = '../sample_data/sample_' + sample + '_anno.h5'
n_features = 4000
data_mat = h5py.File(inputfile)
x = np.array(data_mat['X'])
y0 = np.array(data_mat['Y'])
loc = np.array(data_mat["Pos"])
loc = np.transpose(loc)
data_mat.close()

#remove the spots with NA labels
f = np.where(y0.astype(np.str) != "NA")[0]
x = x[f,:]
loc = loc[f,:]

#build knn graph
A = kneighbors_graph(loc, 30, mode="connectivity", metric="euclidean", include_self=False, n_jobs=-1)
A = A.toarray()

#normalized data
adata0 = sc.AnnData(x)
adata0 = read_dataset(adata0,transpose=False,test_split=False,copy=True) 
adata0 = normalize(adata0,size_factors=True,normalize_input=True,logtrans_input=True)

#sort and extract features by Moran's I
s = np.sum(A)
N = len(adata0.X[:,0])
s1 = N/s
def Iscore_genes(g=np.array([]), A=np.array([[]])):
    g_= np.mean(g)
    g_f = g - g_
    g_f_1 = g_f.reshape(1,-1)
    g_f_2 = g_f.reshape(-1,1)
    A_ = np.dot(g_f_2,g_f_1)
    r = np.sum(A*A_)
    l = np.sum(g_f**2)
    s2 = r/l   
    return s1*s2
    
def getindex(i): 
    if i%100==0:
       print(i)
    gene = adata0.X[:,i]
    I = Iscore_genes(gene, A)
    return I

score = Parallel(n_jobs=64)(delayed(getindex)(i) for i in range(adata0.X.shape[1]))
         
score_ = np.argpartition(score, -n_features)[-n_features:]

#output the index of the HVGs
np.savetxt("sample_" + sample + "_featureSelection_Index4000.csv", score_, delimiter="\t")

