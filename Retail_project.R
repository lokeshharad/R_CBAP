
# Project 2 Retail : Classify shop is opened or not
# Edvancer Eduvantures
# Lokesh Daulat Harad
# 29/05/2018

# ---------------------------------------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(ggplot2)
library(neuralnet)
library(randomForest)
library(gbm)
library(e1071)
library(pROC)
library(caret)
library(cvTools)

# -------------------------------------------------------------------------------------------------------------

# Load DataSets 

store_train<-read.csv("E:\\edvancer\\business_analytics\\r_programming\\Projects\\2_Retail\\store_train.csv",
                      stringsAsFactors = F)
store_test<-read.csv("E:\\edvancer\\business_analytics\\r_programming\\Projects\\2_Retail\\store_test.csv",
                     stringsAsFactors = F)

store_test<-store_test %>%
  mutate(store=NA)

all_data<-rbind(store_train,store_test)                   
summary(all_data)
glimpse(all_data)

# -----------------------------------------------------------------------------------------------------------

# we can substitute mode of variable for categorical variable 

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# -------------------------------------------------------------------------------------------

# Missing values imputation 
# country = mode : 3 & population = median(population)

all_data$country[which(is.na(all_data$country))]<-getmode(all_data$country)         # country
all_data$population[which(is.na(all_data$population))]<-median(all_data$population, # population
                                                               na.rm = T)

# -------------------------------------------------------------------------------------

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}

# -------------------------------------------------------------------------------------------------------

# To check Categorical variables or features

col_names<-names(all_data[sapply(all_data,function(x) is.character(x))])

# -------------------------------------------------------------------------------------------------------

# TO create dummies 
sort(table(all_data$country),decreasing = T)
all_data = CreateDummies(all_data,"country",50)                # Country 

sort(table(all_data$statte_alpha),decreasing = T)
all_data = CreateDummies(all_data,"state_alpha",50)            # state_alpha

all_data[c("State","Id")]<-NULL                                # Discard Id and State

sort(table(all_data$CouSub),decreasing = T)
all_data <- all_data %>%                                       #CouSub 
  mutate(cousub_9999 = as.numeric(all_data$CouSub==99999)) %>%
  select(-CouSub)

sort(table(all_data$countyname),decreasing = T)
all_data = CreateDummies(all_data,"countyname",50)             # countyname

sort(table(all_data$Storecode),decreasing = T)
all_data$storecode[grep("NCNTY+",all_data$storecode)]<-"NCNTY" # storecode 
all_data$storecode[grep("METRO+",all_data$storecode)]<-"METRO"
all_data = CreateDummies(all_data,"storecode",50)

sort(table(all_data$Areaname),decreasing = T)
all_data = CreateDummies(all_data,"Areaname",25)               # Areaname

sort(table(all_data$countytownname),decreasing = T)
all_data = CreateDummies(all_data,"countytownname",15)         # contytownname

sort(table(all_data$store_Type),decreasing = T)
all_data = CreateDummies(all_data,"store_Type",100)            # store_Type

# ------------------------------------------------------------------------------------------------------

# Normalization of Numeric Variables or Feature

all_data<-all_data %>%
  mutate(sales0=as.numeric(scale(all_data$sales0)),         # sales0
         sales1=as.numeric(scale(all_data$sales1)),         # sales1
         sales2=as.numeric(scale(all_data$sales2)),         # sales2
         sales3=as.numeric(scale(all_data$sales3)),         # sales3
         sales4=as.numeric(scale(all_data$sales4)),         # sales4
         population=as.numeric(scale(all_data$population))) # population

# ----------------------------------------------------------------------------------------------

# Seperate training and testing dataset after data cleaning and data preparation
names(all_data)[86]<-"Areaname_BCQMA"
names(all_data)[84]<-"Areaname_PCME_p"
names(all_data)[83]<-"Areaname_Hr_WHr_EHr"
names(all_data)[81]<-"Areaname_SfMA"
names(all_data)[79]<-"Areaname_PFRRi"

prep.store_test<-all_data[which(is.na(all_data$store)),]
prep.store_test<-prep.store_test %>% select(-store)
prep.store_train<-all_data[which(!is.na(all_data$store)),]
prep.store_train$store <-as.factor(prep.store_train$store)

# --------------------------------------
set.seed(123)
train_data<-prep.store_train[sample(1:nrow(prep.store_train),0.8*nrow(prep.store_train)),]
train_label<- prep.store_train[rownames(train_data),"store"]
val_data<-prep.store_train[-sample(1:nrow(prep.store_train),0.8*nrow(prep.store_train)),]
val_label<- prep.store_train[rownames(val_data),"store"]
# -----------------------------------------------------------------------------------------------------
# CROSS VALIDATION AND PARAMETER TUNING
subset_paras = function(full_list_paras,n = 10){
  all_comb = expand.grid(full_list_paras)
  s = sample(1:nrow(all_comb),n)
  subset_para = all_comb[s,]
  return(subset_para)
}

mycost_auc=function(y,yhat){
  cm=confusionMatrix(y,as.factor(yhat))
  score=cm$overall[1]
  return(score)
}

param=list(mtry=c(5,10,15,20,25,35),
           ntree=c(50,100,200,500,700),
           maxnodes=c(5,10,15,20,30,50,100),
           nodesize=c(1,2,5,10)
)

num_trials=50
my_params=subset_paras(param,num_trials)
my_params
f = as.formula(paste("store","~",paste(names(prep.store_train)[-7],collapse = "+")))

myauc=0
for(i in 1:num_trials){
  tictoc::tic()
  print(paste('starting iteration :',i))
  # uncomment the line above to keep track of progress
  params=my_params[i,]
  k=cvTuning(randomForest,store~.,
             data = prep.store_train,
             tuning = params,
             folds = cvFolds(nrow(prep.store_train), K=10, type ="random"),
             cost =mycost_auc, seed =2,
             predictArgs = list(type="prob")
  )
  score.this=k$cv[,2]
  if(score.this>myauc){
    #print(params)
    # uncomment the line above to keep track of progress
    myauc=score.this
    print(myauc)
    # uncomment the line above to keep track of progress
    best_params=params
  }
  #print('DONE')
  tictoc::toc()
  # uncomment the line above to keep track of progress
}

# Best Parameters
# mtry ntree maxnodes nodesize auc
# 35   200      100       10     0.8456
# ---------------------------------------------------------
# FINAL TRAINING WITH BEST PARAMETERS
final_rf_model<-randomForest(store~.,
                             data = prep.store_train,
                             mtry=40,
                             ntree=450,
                             maxnodes=200,
                             nodesize=10,do.trace=T)
final_rf_pred<-as.data.frame(round(predict(final_rf_model,
                                           newdata = prep.store_train,
                                           type = "prob")[,2]))
names(final_rf_pred)[1]<-"store"
cm_rf<-confusionMatrix(as.factor(prep.store_train$store),
                as.factor(final_rf_pred$store))
cm_rf
# Best Parameters
# mtry ntree maxnodes nodesize auc       #Accuracy : 0.1456
# 35   200      100       10    0.8456

 #          Reference
 # Prediction    0    1
 #           0  190 1685
 #           1 1167  296


write.csv(final_rf_pred,"Lokesh_Harad_P2_part2.csv",
                     row.names = F)

# ----------------------------------------------------------

# Support Vrector Machines
svm_model<-svm(store~.,
               data=prep.store_train,
               type="C-classification",
               gamma=0.01,
               kernel="radial",
               cost=1000000,
               scale = T,
               probability=T)
summary(svm_model)
svm_pred<-as.data.frame(predict(svm_model,newdata = prep.store_train))
names(svm_pred)[1]<-"store"
cm_svm<-confusionMatrix(as.factor(svm_pred$store),
                    prep.store_train$store)#Accuracy : 0.9655

roc(svm_pred$store,store_train$store)#Area under the curve: 0.9664

write.csv(svm_pred,"Lokesh_Harad_P2_part2_svm_30_05_18.csv",
          row.names = F)
# --------------------------------------------------------------------------------------

#to find falsely predicted (neg/pos) observations  (svm model predictions)

#           Reference
# Prediction         0    1
#               0 1835   75
#               1   40 1388

x<-prep.store_train$store==0 & svm_pred$store==1 #falsely predicted as FN
FN_observation<-which(x==T)   # we can do directly this (store_train[which(x==T),])
store_train[FN_observation,]  #instead of line 264 and 265
y = prep.store_train$store==1 & svm_pred$store==0 #falsely predicted as FP
FP_observation<-which(y==T)
store_train[FP_observation,]


