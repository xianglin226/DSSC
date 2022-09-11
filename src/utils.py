import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from munkres import Munkres
from scipy import stats, spatial
import random

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]
        
        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.1, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res


def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the predicted clustering labels
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] !=c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def eval_cluster(y_true, y_pred):
    acc = 1-err_rate(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)

    return acc, nmi, ari

def Iscore_gene(y, A):
    s = sum(sum(A))
    N = len(y)
    y_=np.mean(y)
    y_f = y - y_
    y_f_1 = y_f.reshape(1,-1)
    y_f_2 = y_f.reshape(-1,1)
    r = sum(sum(A*np.dot(y_f_2,y_f_1)))
    l = sum(y_f**2)
    s2 = r/l
    s1 = N/s
    return s1*s2

def Iscore_label(y, A):
    s = sum(sum(A))
    N = len(y)
    y_1 = y.reshape(1,-1)
    y_2 = y.reshape(-1,1)
    y_2 = np.reciprocal(y_2)
    z = np.dot(y_2,y_1)
    z[z != 1] = 0
    r = sum(sum(A*z))
    s2 = r/N
    s1 = N/s
    return s1*s2
    
def knn_ACC(p, lab):
    lab_new = []
    for i in range(lab.shape[0]):
        labels = lab[p[i]]
        l_mode = stats.mode(labels).mode[0]
        lab_new.append(l_mode)
    lab_new = np.array(lab_new)
    return sum(lab == lab_new)/lab.shape[0]

###################################################################################
###Some functions to build constraints in python

def generate_random_pair_from_markers(m):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
                if ind1 == l1 and ind2 == l2:
                    return True
        return False

    #net_triu = ssp.triu(A, k=1)
    row, col, _ = ssp.find(A)
    print("Total Links: ", row.shape[0])
    for i in range(row.shape[0]):
        tmp1 = row[i]
        tmp2 = col[i]
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if A[tmp1,tmp2]>0:
           ml_ind1.append(tmp1)
           ml_ind2.append(tmp2)
        else:
            continue

    ml_ind1, ml_ind2 = np.array(ml_ind1), np.array(ml_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    print("Constraints summary: ML=%.0f" % (len(ml_ind1)))
    return ml_ind1, ml_ind2
    
def generate_random_pair_from_label(n,y):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
                if ind1 == l1 and ind2 == l2:
                    return True
        return False

    clu = np.unique(y).tolist()
    for i in clu:
        hit = np.where(y==i)[0]
        k=n
        while k >0:
           cells = np.random.choice(hit,2, replace=False)
           tmp1 = cells[0]
           tmp2 = cells[1]
           if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
               continue
           else:
              ml_ind1.append(tmp1)
              ml_ind2.append(tmp2)
              k-=1
            
    k = n * len(clu)
    while k >0:
        tmp1 = random.randint(0, len(y) - 1)
        tmp2 = random.randint(0, len(y) - 1)
        if tmp1 == tmp2:
           continue
        if check_ind(tmp1, tmp2, cl_ind1, cl_ind2):
           continue
        if y[tmp1] != y[tmp2]:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
            k-=1
        else:
           continue

    ml_ind1, ml_ind2 = np.array(ml_ind1), np.array(ml_ind2)
    cl_ind1, cl_ind2 = np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    print("Constraints summary: ML=%.0f" % (len(ml_ind1)))
    print("Constraints summary: CL=%.0f" % (len(cl_ind1)))
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2
    
def generate_random_pair_from_label2(k1,k2,y):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
                if ind1 == l1 and ind2 == l2:
                    return True
        return False

    while k1 >0 or k2 > 0:
        tmp1 = random.randint(0, len(y) - 1)
        tmp2 = random.randint(0, len(y) - 1)
        if tmp1 == tmp2:
           continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
           continue
        if check_ind(tmp1, tmp2, cl_ind1, cl_ind2):
           continue
        if y[tmp1] == y[tmp2] and k1 > 0:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
            k1-=1
        elif y[tmp1] != y[tmp2] and k2 > 0:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
            k2-=1
        else:
           continue

    ml_ind1, ml_ind2 = np.array(ml_ind1), np.array(ml_ind2)
    cl_ind1, cl_ind2 = np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    print("Constraints summary: ML=%.0f" % (len(ml_ind1)))
    print("Constraints summary: CL=%.0f" % (len(cl_ind1)))
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2
