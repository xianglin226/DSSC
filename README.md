# DSSC
A model-based constrained deep learning clustering approach for spatial-resolved single-cell data

# Packages
Python 3.8.1

Pytorch 1.6.0

Scanpy 1.6.0

SKlearn 0.22.1

Numpy 1.18.1

h5py 2.9.0

munkres 1.1.4  

dgl 0.8.0

All experiments of DSSC in this study are conducted on Nvidia Tesla P100 (16G) GPU.

#The input data should be in h5 format with:  
(1) "X" - count matrix  
(2) "Y" - true labels  
(3) "Pos" - spatial coordinate  
(4) "Genes" - feature names (Use to build constraints)  

# DSSC Procedure  
1) Feature selection (See FeatureSelection.py)  
2) Build constraints (See make_links_fromMakrs.R)  
3) Run DSSC (See run.sh)  
