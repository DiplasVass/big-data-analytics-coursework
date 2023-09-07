# PCA for big data analytics
library(ggplot2)
library(corrplot)
library(GGally)
library(Matrix)
library(ggfortify)
library(dplyr)
library(ggpubr)
library(cluster)    
library(factoextra) 
library(gridExtra)
library(rpart)
library(RColorBrewer)
library(party)
library(randomForest)
library(tree)
library(caret)
library(ROCR)

# Import the dataset:

df <- read.csv("C://Users//basil//OneDrive//Έγγραφα//brunelwin//BIG DATA ANALYTICS//heptatlon.csv",
               header = TRUE)
df

if(is.data.frame(df)==FALSE) {
  df = as.data.frame(df)
} else {df}

# Class of variables:
sapply(df,class) # 1 variable is character, 7 numerics and 1 integer

# Dimension of data:
dim(df) # 25 rows and 9 columns
25*9 # 225 saved data points in heptatlon

# Names of variables:
data.frame('Names of Variables' = colnames(df)) # Names of variables

# Check for missing values:
which(is.na(df)==TRUE) # 0 NAs in data

# Summary of data:
summary(df)

# Data frame of sports variables
df.1 <- df %>% select(-X,-score)
dim(df.1)

# Correlation and density plots:
ggscatmat(df.1,columns = 1:ncol(df.1))

res <- round(cor(df.1),2)
corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

# We standardize the data in order to transfer all the 
# variables in the same units of measure 

St <- matrix(0,25,7)

M_i <- apply(df.1,2,mean) # mean at each column
s_i <- apply(df.1,2,sd)   # std at each column

for(i in 1:7){
  St[,i]  = (df.1[,i] - M_i[i])/s_i[i]
}

St # Standardized data 

X <- cov(St) # note the cov() gives the same outcome with cor()
             # because we have standardized the data.
cor(St)

rankMatrix(X)[1] # The rank of the matrix is 7 = variables. (full ranked)

#-----------------------------------------#
#------------------- PCA -----------------#
#-----------------------------------------#

# We start the PCA

# We find the eigenvalues and eigenvectors with the command (eigen):

# Eigenvalues:
lamda <- eigen(X)$values
lamda 

# We find the mean of eigenvalues
lamda.bar <- mean(lamda)
lamda.bar

which(lamda>lamda.bar) # We choose to work with the first two eigenvalues:
prop <- lamda/sum(lamda)
cumsum(prop) # Cumulative proportion of eigenvalues

# Scree plot:

index <- 1:length(lamda)
lamda

df.lamda <- data.frame(index,lamda,'cdf' =  cumsum(prop))

g1 <- df.lamda %>%
    ggplot(aes(x=index,y=lamda))+
  geom_point(size=2,colour='black') +
  geom_line(colour='red') +
  geom_hline(yintercept=1,size=0.5,linetype='dotted') +
  annotate("text", x = 5, y=1,label = "Lambda bar = 1", vjust = -0.5)+
  labs(y = 'Eigenvalues')+
  ggtitle('Scree plot') +
  theme_bw()

g2 <- df.lamda %>%
  ggplot(aes(x=index,y=cdf))+
  geom_point(size=2,colour='black') +
  geom_line(colour='red') +
  geom_hline(yintercept=0.85,size=0.5,linetype='dotted') +
  annotate("text", x = 5, y=0.85,label = "Determined proportion = 0.85 (85%)", vjust = -0.5)+
  labs(y = 'Eigenvalues')+
  ggtitle('Cumsum of proportion of eigenvalues') +
  theme_bw()

ggarrange(g1,g2,nrow = 1,ncol=2)

# Eigenvectors:
T <- eigen(X)$vectors
T

# Eigenvalues with the names of variables:
vec <- data.frame(colnames(df)[-c(1,9,10,11)],T)
colnames(vec) <- c('names','T1','T2','T3','T4','T5','T6','T7')
vec

# Scores:
# From theory we know that Z = X*T

Z <- St%*%T

Z1 <- Z[,1] # The score of the first principal
Z2 <- Z[,2] # The score of the second principal
Z1
Z2

plot(Z1,Z2)
plot(T[,1],T[,2])

cov(Z)
sum(diag(cov(Z)))
sum(diag(X))
sum(lamda)

# The first two eigenvectors:
vec[,1:3]

# Adding the scores in data:  
df[,10] = Z1
cor(df$score,df$Z1)

df[,11] = Z2
colnames(df)[10:11] = c('Z1','Z2')

# Graph of scores: 
df %>%  
  ggplot(aes(x=Z1,y=Z2,group = X)) +
  geom_point() +
  labs(x = 'PC1 (63.72%)', y = 'PC2 (17.06%)',
       title = 'Scatter-plot of the two scores') +
  theme_bw()

# Biplot:

df.2 <- df %>% select(-score,-X,-Z1,-Z2)
pca_res <- prcomp(df.2, scale. = TRUE)
autoplot(pca_res,df,loadings = TRUE, loadings.colour = 'blue',
         loadings.label = TRUE)+
  labs(title = 'Biplot of the two scores') +
  theme_bw()

# Graph of eigenvectors:
ggplot(vec,aes(x = T1, y = T2, group = names,color = names)) +
  geom_point() +
  labs(x = 'T1 (first eigenvector)', y = 'T2 (second eigenvector)',
       title = 'Plot of the first two eigenvectors')+
  theme_bw()

#-----------------------------------------#
#----------- K-mean Clustering -----------#
#-----------------------------------------#

scores <- df %>% select(Z1,Z2)

kmeans(scores,3)$cluster
library(cluster)
install.packages('factoextra')
library(factoextra)

fviz_nbclust(scores, kmeans, method = 'wss')
fviz_nbclust(scores, kmeans, method = 'silhouette')
fviz_nbclust(scores, kmeans, method = 'gap_stat')

k = 3
kmeans_df = kmeans(scores,iter.max=50,centers = k,nstart=50)
fviz_cluster(kmeans_df, data = scores)

# Index 25 is an outlier, we exclude from the data and we have 2 clusters
df[25,]
kmeans_df = kmeans(scores[-25,],iter.max=50,centers = 2,nstart=50)
fviz_cluster(kmeans_df, data = scores[-25,])

#------------------------------------------#
#------------ Cluster Analysis ------------#
#------------------------------------------#

df <- read.csv("C://Users//basil//OneDrive//Έγγραφα//brunelwin//BIG DATA ANALYTICS//protein.csv",header = TRUE)

# We exclude the column of Countries and we make a data frame without countries
# with row names the countries.

row.names(df) <- df$Country
df <- data.frame(df %>% select(-Country))
df

head(df,5)
str(df)

# No missing values
is.na(df)

# Summary of data:
summary(df)

# Correlation and Density plot:
ggscatmat(df,columns = 1:ncol(df))

# Boxplots of data:
df %>%  boxplot(col=rainbow(9),main='Boxplots of data',xaxt='n',pch=20)
          text(1,23,'RedMeat')
            text(2,18,'WhiteMeat')
              text(3,8,'Eggs')
                text(4,37,'Milk')
                  text(5,17,'Fish')
                    text(6,58,'Cereals')
                      text(7,10,'Starch')
                        text(8,11,'Nuts')
                          text(9,12,'Fr.Veg')

# From the boxplots we detect some outliers. 

## Detecting Outliers:

# Make a loop for detecting the index of outliers: 

v = integer(0)
for(i in 1:ncol(df)){
vs = which(df[,i] %in% boxplot(df[,i],plot = FALSE)$out)
v = unique(c(vs,v))
}
v

# Now we exclude them from the data:
df <- df[-v,]

# Normalize the data:

standard = function(x) {
  if(is.numeric(x) == TRUE) {
   return((x - min(x))/(max(x)-min(x)))
                            }
  else if(is.character(x) == TRUE){return(x)}
                       }

df_st <- data.frame(df %>% apply(2,standard))

#-----------------------------------#
#-----Agglomerative Clustering------#
#-----------------------------------#

# Complete method:

df_st  %>% 
   dist(method = 'euclidean') %>% 
     hclust(method = 'complete') %>%
       plot(hang = -0.1,
        main = 'Cluster Dendrogram with complete method') 
          rect.hclust(df_st %>% 
              dist(method = 'euclidean') %>% 
                hclust(method = 'complete'),
                  k=5)

### Select a partition containing k=5 groups

clusters_5 <- df_st  %>% 
                dist(method = 'euclidean') %>% 
                  hclust(method = 'complete') %>%
                    cutree(k = 5)

clusters_5

#---------------------#
#------ K-Means ------#
#---------------------#

# We choose k = 5 for kmeans clustering.

kmean_cluster_5 <- (df_st %>% kmeans(centers=5,nstart = 25))$cluster

# Evaluation of cluster results:
# Here we will use the Silhouette method in order to examine which cluster 
# method is better.

# Silhouette method for Hierarchical Clustering with 5 partitions:

sil_hc_5      <- silhouette(clusters_5,
                    df_st %>% dist(method = 'euclidean'))
row.names(sil_hc_5) <- row.names(df_st)
sil_hc_5

# Silhouette method for K-Means Clustering with 5 partitions:

sil_k_means_5 <- silhouette(kmean_cluster_5,
                              df_st %>% dist(method = 'euclidean'))

row.names(sil_k_means_5) <- row.names(df_st)
sil_k_means_5

# Silhouette ggplots:

s1 <- fviz_silhouette(sil_hc_5,label = TRUE,rect = TRUE, palette = "jco")
s1$layers[[2]]$aes_params$colour <- "black"
s1

s2 <- fviz_silhouette(sil_k_means_5,label = TRUE,rect = TRUE, palette = "jco")
s2$layers[[2]]$aes_params$colour <- "black"
s2

# Plotting the clusters with kmeans method with the help of PCA:

p1 <-    fviz_cluster(df_st %>% 
                        kmeans(centers=2,nstart = 25) ,
                         data = df_st , palette = "jco" ) + 
        ggtitle("k = 2") + theme_bw()

p2 <-    fviz_cluster(df_st  %>% 
                         kmeans(centers=3,nstart = 25) ,
                          data = df_st  , palette = "jco") + 
        ggtitle("k = 3") + theme_bw()

p3 <-    fviz_cluster(df_st %>% 
                        kmeans(centers=4,nstart = 25) ,
                         data = df_st  ,  palette = "jco") + 
        ggtitle("k = 4") + theme_bw()

p4 <-    fviz_cluster(df_st %>% 
                        kmeans(centers=5,nstart = 25) ,
                         data = df_st , palette = "jco") + 
        ggtitle("k = 5") + theme_bw()

grid.arrange(p1, p2, p3, p4, nrow = 2) 

# We observe that k = 3 is a good choice.

# Optimal number of clusters for kmeans method:

fviz_nbclust(df_st, kmeans, method = "silhouette")

# We conduct that k = 3 is an optima number of partitions.

#--------------------------------
# Data: Reading Skills
#--------------------------------

#--------------------------------#
#-------- Decision Trees --------#
#--------------------------------#
# Split the data into training and testing (75%-25%):

df <- readingSkills
dim(df)
summary(df)

set.seed(52)
ind <- sample(1:dim(df)[1],200*(0.75))
ind

training.df <- df[ind,]
testing.df  <- df[-ind,]

# Decision trees:
formula     <- nativeSpeaker ~ age + shoeSize + score
output.tree <- tree(formula = formula, data = training.df)
 
# Plot of the tree:
plot(output.tree)
text(output.tree, pretty = 1)

summary(output.tree)

# Cross validation:
cv_output.tree <- cv.tree(output.tree, FUN=prune.misclass)

cv_table <- data.frame(
  size  = cv_output.tree$size,
  error = cv_output.tree$dev
)

pruned_tree_size <- cv_table[which(cv_table$error == min(cv_table$error)),'size']

# prune the tree to the required size:
pruned_tree_df <- prune.misclass(output.tree,best=pruned_tree_size)

# plot
plot(pruned_tree_df)
text(pruned_tree_df,pretty=0)

# Comparing unpruned and pruned trees:
par(mfrow=c(2,1))
plot(output.tree)
text(output.tree, pretty = 1)
plot(pruned_tree_df)
text(pruned_tree_df,pretty=0)
par(mfrow = c(1,1))

######################################################################
# 4. Decision tree prediction

predict_output.tree <- predict(output.tree,testing.df[,-1],type='class')
predict_output.tree

pruned_predict_output.tree <- predict(pruned_tree_df,testing.df[,-1],type='class')
pruned_predict_output.tree

predict_tree_table <- data.frame(actual   = testing.df$nativeSpeaker,
                                 unpruned = predict_output.tree,
                                 pruned   = pruned_predict_output.tree)

predict_tree_table

unpruned_tree_table <- table(predict_tree_table[,c('actual','unpruned')])
unpruned_tree_table

pruned_tree_table <-  table(predict_tree_table[,c('actual','pruned')])
pruned_tree_table

#Evaluation:

#Confusion matrix for predictions of decision tree algorithm:
tree_conf_matrix <- 
confusionMatrix(data = predict_output.tree,reference = testing.df$nativeSpeaker)
tree_conf_matrix

#-------------------------------
# Random Forests
#-------------------------------

# apply random forest to data:
rf_df <- randomForest(nativeSpeaker ~ age + shoeSize + score,ntree=500,
                      importance = T,data=training.df)

# Plot of random forest:
plot(rf_df)
legend('topright', colnames(rf_df$err.rate), bty = 'n', lty = c(1,2,3), col = c(1:3))

# variable importance:
varImpPlot(rf_df, type = 1)

# Predict:

rf_df_predict <- predict(rf_df,testing.df[,-1],type = 'class')

rf_predict_results <- data.frame(actual = testing.df$nativeSpeaker,predict=rf_df_predict)
rf_predict_results

rf_df_predict_table <- table(rf_predict_results)
rf_df_predict_table

# Confusion matrix:

rf_conf_matrix <- 
  confusionMatrix(data = rf_df_predict,reference = testing.df$nativeSpeaker)
rf_conf_matrix

# ROC curve for decision trees and random forest:

# set the parameters for tuning to 10-fold CV
ctrl_parameters <- trainControl(method = 'CV', number = 10)

# Train a tree:
train_tree <- train(formula,data=training.df,method='rpart', trControl = ctrl_parameters)
train_tree

# Train a forest:
train_rf  <- train(formula,data=training.df,method='rf', trControl = ctrl_parameters)
train_rf

pred_tree <- predict(train_tree,testing.df[,-1],type='prob')$yes
pred_rf   <- predict(train_rf,testing.df[,-1],type='prob')$yes

# data frame containing probability values for the prediction 'yes' and 
# data frame for which a yes is true and no is false.

models_prob <- data.frame(tree = pred_tree, rf = pred_rf)
models_label <- data.frame(tree = testing.df$nativeSpeaker =='yes', 
                           rf = testing.df$nativeSpeaker == 'yes')

# ROC curves:
ROC_pred <- prediction(models_prob,models_label)
ROC_perf <- performance(ROC_pred,'tpr','fpr')

plot(ROC_perf,col=as.list(c('red','blue')),
main='ROC curves for decision tree and random forest algorithms')
abline(a = 0, b = 1, lty = 2, col = 'black')
legend(
  "bottomright",
  names(models_prob),
  col = c("red", "blue"),
  lty = rep(1,2),
  bty = 'n')

# Random forest algorithm is better classification algorithm than decision tree
# for this data set according to ROC curve.

#--------------------------------
# Data: Carseats
#--------------------------------
df <- read.csv("C://Users//basil//Downloads//carseats_sale.csv",stringsAsFactors = T)

#--------------------------------#
#-------- Decision Trees --------#
#--------------------------------#

# Summary of data:
dim(df)
summary(df)
head(df,4)

# Split the data into training and testing (75%-25%):
set.seed(52)
ind <- sample(1:dim(df)[1],dim(df)[1]*(0.75))
ind

training.df <- df[ind,]
testing.df  <- df[-ind,]

# Decision trees:
formula     <- reformulate(names(training.df[, -1]), response = 'Sales') 
output.tree <- tree(formula = formula, data = training.df)

# Plot of the tree:
plot(output.tree)
text(output.tree, pretty = 1)

summary(output.tree)

# Cross validation:
cv_output.tree <- cv.tree(output.tree, FUN=prune.misclass)

cv_table <- data.frame(
  size  = cv_output.tree$size,
  error = cv_output.tree$dev
)

pruned_tree_size <- cv_table[which(cv_table$error == min(cv_table$error)),'size']

# prune the tree to the required size:
pruned_tree_df <- prune.misclass(output.tree,best=pruned_tree_size)

# plot
plot(pruned_tree_df)
text(pruned_tree_df,pretty=0)

# Comparing unpruned and pruned trees:
par(mfrow=c(2,1))
plot(output.tree)
text(output.tree, pretty = 1)
plot(pruned_tree_df)
text(pruned_tree_df,pretty=0)
par(mfrow = c(1,1))

######################################################################
# 4. Decision tree prediction

predict_output.tree <- predict(output.tree,testing.df[,-1],type='class')
predict_output.tree

pruned_predict_output.tree <- predict(pruned_tree_df,testing.df[,-1],type='class')
pruned_predict_output.tree

predict_tree_table <- data.frame(actual   = testing.df$Sales,
                                 unpruned = predict_output.tree,
                                 pruned   = pruned_predict_output.tree)

predict_tree_table

unpruned_tree_table <- table(predict_tree_table[,c('actual','unpruned')])
unpruned_tree_table

pruned_tree_table <-  table(predict_tree_table[,c('actual','pruned')])
pruned_tree_table

#Evaluation:
#Confusion matrix for predictions of decision tree algorithm:
tree_conf_matrix <- 
  confusionMatrix(data = predict_output.tree,reference = testing.df$Sales)
tree_conf_matrix

#-------------------------------
# Random Forests
#-------------------------------

# apply random forest to data:
rf_df <- randomForest(formula,ntree=500,
                      importance = T,data=training.df)

# Plot of random forest:
plot(rf_df)
legend('topright', colnames(rf_df$err.rate), bty = 'n', lty = c(1,2,3), col = c(1:3))

# variable importance:
varImpPlot(rf_df, type = 1)

# Predict:

rf_df_predict <- predict(rf_df,testing.df[,-1],type = 'class')

rf_predict_results <- data.frame(actual = testing.df$Sales,predict=rf_df_predict)
rf_predict_results

rf_df_predict_table <- table(rf_predict_results)
rf_df_predict_table

# Confusion matrix:

rf_conf_matrix <- 
  confusionMatrix(data = rf_df_predict,reference = testing.df$Sales)
rf_conf_matrix

# ROC curve for decision trees and random forest:

# set the parameters for tuning to 10-fold CV
ctrl_parameters <- trainControl(method = 'CV', number = 10)

# Train a tree:
train_tree <- train(formula,data=training.df,method='rpart', trControl = ctrl_parameters)
train_tree

# Train a forest:
train_rf  <- train(formula,data=training.df,method='rf', trControl = ctrl_parameters)
train_rf

pred_tree <- predict(train_tree,testing.df[,-1],type='prob')$High
pred_rf   <- predict(train_rf,testing.df[,-1],type='prob')$High

# data frame containing probability values for the prediction 'yes' and 
# data frame for which a yes is true and no is false.

models_prob <- data.frame(tree = pred_tree, rf = pred_rf)
models_label <- data.frame(tree = testing.df$Sales =='High', 
                           rf = testing.df$Sales == 'High')

# ROC curves:
ROC_pred <- prediction(models_prob,models_label)
ROC_perf <- performance(ROC_pred,'tpr','fpr')

plot(ROC_perf,col=as.list(c('red','blue')),
     main='ROC curves for decision tree and random forest algorithms')
abline(a = 0, b = 1, lty = 2, col = 'black')
legend(
  "bottomright",
  names(models_prob),
  col = c("red", "blue"),
  lty = rep(1,2),
  bty = 'n')

# Random forest algorithm is better classification algorithm than decision tree
# for this data set according to ROC curve.
