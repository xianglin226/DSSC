import scanpy as sc
import h5py
import numpy as np
from utils import *
from preprocess0 import read_dataset, normalize
from sklearn.neighbors import kneighbors_graph

#x is the count matrix; 
#Loc is the spatial coordinates
#Here we use the sample 151507 as an example, y is just used to remove unknown cells
sample = "151507"
n_features = 2000
data_mat = h5py.File('./sample_' + sample + '_anno.h5')
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
score = []
for i in range(adata0.X.shape[1]):
      gene = adata0.X[:,i]
      I = Iscore_gene(gene, A)
      score.append(I)
      if i%100==0:
         print(i)            
score_ = np.argpartition(score, -n_features)[-n_features:]

#output the index of the HVGs
np.savetxt("sample_" + sample + "_featureSelection_Index2000.csv", score_, delimiter="\t")
