import pickle, os, numbers

import numpy as np
import scipy
from scipy.sparse import issparse
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.preprocessing import StandardScaler
import sklearn


CHUNK_SIZE = 20000


def pearson_residuals(counts, theta, clipping=True):
    '''Computes analytical residuals for NB model with a fixed theta, clipping outlier residuals to sqrt(N)'''
    counts_sum0 = np.sum(counts, axis=0, keepdims=True)
    counts_sum1 = np.sum(counts, axis=1, keepdims=True)
    counts_sum  = np.sum(counts)

    #get residuals
    mu = counts_sum1 @ counts_sum0 / counts_sum
    z = (counts - mu) / np.sqrt(mu + mu**2/theta)

    #clip to sqrt(n)
    if clipping:
        n = counts.shape[0]
        z[z >  np.sqrt(n)] =  np.sqrt(n)
        z[z < -np.sqrt(n)] = -np.sqrt(n)
    
    return z


def concat_data(
        adata_list, 
        batch_categories=None, 
        join='inner',             
        batch_key='batch', 
        index_unique=None, 
        save=None
    ):
    """
    Concatenate multiple datasets along the observations axis with name ``batch_key``.
    
    Parameters
    ----------
    adata_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a “batch”.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. Default: 'inner'.
    batch_key
        Add the batch annotation to obs using this key. Default: 'batch'.
    index_unique
        Make the index unique by joining the existing index names with the batch category, using index_unique='-', for instance. Provide None to keep existing indices.
    save
        Path to save the new merged AnnData. Default: None.
        
    Returns
    -------
    New merged AnnData.
    """

    if batch_categories is None:
        batch_categories = list(map(str, range(len(adata_list))))
    else:
        assert len(adata_list) == len(batch_categories)
    [print(b, adata.shape) for adata,b in zip(adata_list, batch_categories)]
    concat = AnnData.concatenate(*adata_list, join=join, batch_key=batch_key,
                                batch_categories=batch_categories, index_unique=index_unique)  
    if save:
        concat.write(save, compression='gzip')
    return concat


def preprocessing_rna(
        adata: AnnData, 
        min_features: int = 0, 
        min_cells: int = 0, 
        target_sum: int = 10000, 
        n_top_features = 2000, # or gene list
        chunk_size: int = CHUNK_SIZE,
        log=None
    ):
    """
    Preprocessing single-cell RNA-seq data
    
    Parameters
    ----------
    adata
        An AnnData matrice of shape n_obs × n_vars. Rows correspond to cells and columns to genes.
    min_features
        Filtered out cells that are detected in less than n genes. Default: 600.
    min_cells
        Filtered out genes that are detected in less than n cells. Default: 3.
    target_sum
        After normalization, each cell has a total count equal to target_sum. If None, total count of each cell equal to the median of total counts for cells before normalization.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.
    log
        If log, record each operation in the log file. Default: None.
        
    Return
    -------
    The AnnData object after preprocessing.
    """
    if min_features is None: min_features = 600
    if n_top_features is None: n_top_features = 2000
    
    if log: log.info('Preprocessing')
#    if not issparse(adata.X):
#        adata.X = scipy.sparse.csr_matrix(adata.X)
    
    adata = adata[:, [gene for gene in adata.var_names 
                  if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]
    
    if log: log.info('Filtering cells')
    sc.pp.filter_cells(adata, min_genes=min_features)

    count_X = adata.X

    if log: log.info('Filtering features')
    sc.pp.filter_genes(adata, min_cells=min_cells)

#    if log: log.info('Normalizing total per cell')
#    sc.pp.normalize_total(adata, target_sum=target_sum)

    if log: log.info('Normalizing library size per cell')
    sc.pp.normalize_per_cell(adata)
    adata.obs['size_factors'] = adata.obs.n_counts

    if log: log.info('Log1p transforming')
    sc.pp.log1p(adata)

    if log: log.info('Finding variable features')
    if type(n_top_features) == int and n_top_features>0:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key='batch', inplace=False, subset=True)
    elif type(n_top_features) != int:
        adata = reindex(adata, n_top_features)

    high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
    count_X = count_X[:, high_variable]

    #if log: log.info('Batch specific maxabs scaling')
    #adata = batch_scale(adata, chunk_size=chunk_size)
    if log: log.info('Processed dataset shape: {}'.format(adata.shape))
    return adata, count_X


def reindex(adata, genes, chunk_size=CHUNK_SIZE):
    """
    Reindex AnnData with gene list
    
    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing
    chunk_size
        chunk large data into small chunks
        
    Return
    ------
    AnnData
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))
    new_X = scipy.sparse.csr_matrix((adata.shape[0], len(genes)))
    for i in range(new_X.shape[0]//chunk_size+1):
        new_X[i*chunk_size:(i+1)*chunk_size, idx] = adata[i*chunk_size:(i+1)*chunk_size, genes[idx]].X
    adata = AnnData(new_X, obs=adata.obs, var={'var_names':genes}) 
    return adata


def batch_scale(adata, chunk_size=CHUNK_SIZE):
    """
    Batch-specific scale data
    
    Parameters
    ----------
    adata
        AnnData
    chunk_size
        chunk large data into small chunks
    
    Return
    ------
    AnnData
    """
    for b in adata.obs['batch'].unique():
        idx = np.where(adata.obs['batch']==b)[0]
        scaler = StandardScaler(copy=False, with_mean=True).fit(adata.X[idx])
        for i in range(len(idx)//chunk_size+1):
            adata.X[idx[i*chunk_size:(i+1)*chunk_size]] = scaler.transform(adata.X[idx[i*chunk_size:(i+1)*chunk_size]])

    return adata


def read_genelist(filename):
    genelist = list(set(open(filename, 'rt').read().strip().split('\n')))
    assert len(genelist) > 0, 'No genes detected in genelist file'
    print('### Autoencoder: Subset of {} genes will be denoised.'.format(len(genelist)))

    return genelist

def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')

# Shared nearest neighbor graph
def SNN_adj(A):
    Am = A.copy()
    indices = np.split(A.indices, A.indptr)[1:-1]

    for i in range(A.shape[0]):
        for j in indices[i]:
            if A[j, i] == 0:
                Am[i, j] = 0

    Am.eliminate_zeros()
    return Am

def read_pickle(inputfile):
    return pickle.load(open(inputfile, "rb"))

def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if scipy.sparse.issparse(A):
        D = scipy.sparse.diags(degrees)
    else:
        D = np.sparse.diag(degrees)
    return D

def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output