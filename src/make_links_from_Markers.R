library(ggplot2)
library(Seurat)
library(rhdf5)
library(openxlsx)
library(cccd)
sample <- "151507"
dat <- H5Fopen(paste0("./sample_",sample,"_anno.h5"))

##read data
#X is count matrix; Y is the label; Pos is the spatial coordinates.
y <- dat$Y
genes <- dat$Gene
x <- dat$X
rownames(x) <- genes
pos <- dat$Pos

#Filter out the spots with NA labels
f <- is.na(y)
x <- x[,!f]
y <- y[!f]
pos <- pos[!f,]

#build knn graph
k <- nng(as.data.frame(pos),k=6)
knn <- as_adjacency_matrix(k)

#normalize counts
x_n <- NormalizeData(x) # can be normalized by other methods.

#extract the marker genes
#This example is for 151507;
#Marker genes are from the paper of spatialLIBD; more marker genes can be added according to the prior knowledge.
genelist = c("FABP7","PCP4","MOBP","AQP4")
#Layers represented by each marker (or celltypes for other datasets)
layers = c("Layer1","Layer5","WM","Layer1")

#make link candidates, the cutoff value (0.9 and 0.5) can be changed.
cans <- data.frame()
for (k in 1:length(genelist)) {
  gene <- genelist[k]
  g <- x_n[gene,]
  g_ <- c()
  for (i in 1:length(g)) {
    ind <- c(i,which(knn[i,]==1))
    g_[i] <- mean(g[ind])
  } # genes are smoothed by the spatially neighbor cells
  cutoff <- quantile(g_, 0.90) # cutoff 1
  hit <- ifelse(g_>cutoff,1,0)
  rate <- c()
  for (i in 1:nrow(pos)) {
    neb <- knn[i,]
    hit_ <- hit[neb==1]
    rate[i] <- sum(hit_)/length(hit_)
  }
  hit2 <- ifelse(rate>0.5,1,0) #cutoff 2
  if(sum(hit2)==0){
    next
  }
  df <- data.frame(Cells = which(hit2>0), Group=layers[k])
  cans <- rbind(cans,df)
}

#write.table(cans,paste0("LinksFromMarks_",sample,".txt"), col.names = F,row.names = F,quote = F)

#make must-link and cannot-link
ml1 <- c()
ml2 <- c()
cl1 <- c()
cl2 <- c()
n_ml <- 20000
n_cl <- 20000
lim = 1000000 #maximum limitation
while(lim>0 & (n_ml > 0 | n_cl >0)){
  cell1 = sample(1:nrow(cans),1)
  cell2 = sample(1:nrow(cans),1)
  if(cell1 == cell2){next}
  if(cans[cell1,2]==cans[cell2,2] & n_ml > 0){
    ml1 <- c(ml1,cans[cell1,1])
    ml2 <- c(ml2,cans[cell2,1])
    n_ml = n_ml-1}
  if(cans[cell1,2]!=cans[cell2,2] & n_cl > 0){
    cl1 <- c(cl1,cans[cell1,1])
    cl2 <- c(cl2,cans[cell2,1])
    n_cl = n_cl-1}
  else{
    lim = lim-1
    next}
}

ml_df <- data.frame(ml1,ml2)
cl_df <- data.frame(cl1,cl2)
ml_df <- t(apply(ml_df, 1, sort))
cl_df <- t(apply(cl_df, 1, sort))
ml_df <- ml_df[!duplicated(ml_df),]
cl_df <- cl_df[!duplicated(cl_df),]

#check the accuracy of constraints when the true labels are available
if(!is.null(y)){
   check_labels1 <- function(cells){
     sum(y[cells[1]]==y[cells[2]])
     }

   check_labels2 <- function(cells){
     sum(y[cells[1]]!=y[cells[2]])
     }

   print(sum(apply(ml_df, 1, check_labels1))/nrow(ml_df))
   print(sum(apply(cl_df, 1, check_labels2))/nrow(cl_df))
}
write.table(ml_df,paste0("sample_",sample,"_mlFromMarks.txt"), col.names = F,row.names = F,quote = F)
write.table(cl_df,paste0("sample_",sample,"_clFromMarks.txt"), col.names = F,row.names = F,quote = F)
