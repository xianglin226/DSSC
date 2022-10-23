library(ggplot2)
library(Seurat)
library(rhdf5)
library(cccd)

##read data
#load data prepared in h5 format
dat <- H5Fopen("../sample_data/sample_151507_anno.h5")

#X is count matrix; Y is the true label (used for remove NA cells); Pos is the spatial coordinates.
y <- dat$Y
genes <- dat$Gene
x <- dat$X
rownames(x) <- genes
pos <- dat$Pos

#Filter out the spots with NA labels (provided by the author; this is only for the spatialLIBD data)
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
#This example is for spatialLIBD 151507;

genelist = c("FABP7","PCP4","MOBP","AQP4")
#Layers represented by each marker (or celltypes for other datasets)
layers = c("Layer1","Layer5","WM","Layer1")
#whether to get high (1) or low (2) expressed cells
att = c(1,1,1,1)

#Marker genes are from the paper of spatialLIBD
#such as PCP4 (Layer5), MOBP (WM), AQP4 (Layer1), FABP4 (Layer1), CARTPT(Layer3), ENC1(WN), KRT17(Layer6),...
#This is a flexible approach, more marker genes can be added according to the prior knowledge.
#the spatial dependency can be tested by using the function below, it is not suggested to use a gene with low spatial dependency.
#It is also suggested to check the distribution of gene before using as markers (see the codes at bottom).
######################################################################################
##############this is the code to check the spatial dependency of genes###############
knn1 <- nng(pos,k=6)
knn1 <- as_adjacency_matrix(knn1)
knn1 <- as.matrix(knn1)

spatialgenes <- function(g, w){
  s = sum(w)
  N = length(g)
  g_ = mean(g)
  g_f = g - g_
  r = sum(w * (g_f%*%t(g_f)))
  l = sum(g_f^2) 
  s2 = r/l
  s1 = N/s
  return(s1*s2)
}

for (i in genelist) {
  sdscore = spatialgenes(x_n[i,], knn1)
  print(paste(i,sdscore))
}

############################################################
#make link candidates, the cutoff value (0.95 and 0.5) can be changed to adjust the number of candidates.
#there is a tradeoff between the coverage and the correctness of constraints. 
cutoff1 = 0.95
cutoff2 = 0.5
cans <- data.frame()
for (k in 1:length(genelist)) {
  gene <- genelist[k]
  g <- x_n[gene,]
  g_ <- c()
  for (i in 1:length(g)) {
    ind <- c(i,which(knn[i,]==1))
    g_[i] <- mean(g[ind])
  } # genes are smoothed by the spatially neighbor cells
  if(att[k]==1){ # get high (1) or low (2) expression cells
    cutoff <- quantile(g_, cutoff1) #cutoff1
    hit <- ifelse(g_>cutoff,1,0)} 
  else{
    cutoff <- quantile(g_, 1-cutoff1) #cutoff1
    hit <- ifelse(g_<cutoff,1,0)
  }
  
  rate <- c()
  for (i in 1:nrow(pos)) {
    neb <- knn[i,]
    hit_ <- hit[neb==1]
    rate[i] <- sum(hit_)/length(hit_)
  }
  hit2 <- ifelse(rate>cutoff2,1,0) #cutoff 2
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
n_ml <- 40000 #a ml pool
n_cl <- 40000 #a cl pool
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

write.table(ml_df,"sample_151507_mlFromMarks.txt", col.names = F,row.names = F,quote = F)
write.table(cl_df,"sample_151507_clFromMarks.txt", col.names = F,row.names = F,quote = F)

###########################################################################
##################check the pattern of smoothed gene expression#######
#use PCP4 as example
g <-x_n["PCP4",]

#plot1 normalized expression
ggplot(as.data.frame(pos), mapping = aes(V1,V2,color=g)) +
  geom_point() +
  scale_color_gradient(low = "white",high = "red") +
  theme_bw()

#plot2 smoothed expression
g_ <- c()
for (i in 1:length(g)) {
  ind <- c(i,which(knn[i,]==1))
  g_[i] <- mean(g[ind])
}

ggplot(as.data.frame(pos), mapping = aes(V1,V2,color=g_)) +
  geom_point() +
  scale_color_gradient(low = "white",high = "red") +
  theme_bw()

#plot3 expression cutoff filtered expression
cutoff <- quantile(g_, 0.9)
hit <- ifelse(g_>cutoff,1,0)
ggplot(as.data.frame(pos), mapping = aes(V1,V2,color=hit)) +
  geom_point() +
  scale_color_gradient(low = "white",high = "red") +
  theme_bw()

#plot4 spatial neighbor cutoff filtered expression
rate <- c()
for (i in 1:nrow(pos)) {
  neb <- knn[i,]
  hit_ <- hit[neb==1]
  rate[i] <- sum(hit_)/length(hit_)
}

hit2 <- ifelse(rate>0.5,1,0)
ggplot(as.data.frame(pos), mapping = aes(V1,V2,color=hit2)) +
  geom_point() +
  scale_color_gradient(low = "white",high = "red") +
  theme_bw()
