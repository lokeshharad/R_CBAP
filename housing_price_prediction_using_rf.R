# -----------------------------------------------##########-------------------------------------------------

## Name of Project : Predict a House Price using Predictive Model(Linear Regression) 
## Project Member  : Lokesh Daulat Harad
## Date : 15 May 2018
## Edvancer Edvantures

#------------------------------------------------##########--------------------------------------------------

# Load packages 
library(psych)
library(dplyr)
library(tidyr)
library(ggplot2)
library(randomForest)
library(cvTools)
library(tictoc)
library(e1071)
library(caret)
library(gbm)
#----------------------------------------------------------------------------------------------------------------

# Load a Training dataset 
house_train <- read.csv("D:\\edvancer\\business_analytics\\r_programming\\Projects\\1_Real_Estate\\housing_train.csv",
                        stringsAsFactors = F)
glimpse(house_train)


#  Here we create ID as feature to identify Training dataset 
house_train <- house_train %>% 
  mutate(ID = "Train")
# ---------------------------------------------------------------

# Load a Testing dataset
house_test <- read.csv("D:\\edvancer\\business_analytics\\r_programming\\Projects\\1_Real_Estate\\housing_test.csv",stringsAsFactors = F)
glimpse(house_test)

# We add two features or variables ID and Price 
# because below we create all_data dataset for assinging new features 
#  or dummy variables from old variales like Suburb, Type, Method,etc.

house_test <- house_test %>% 
  mutate(Price = NA,
         ID    = "Test")

# ----------------------------------------------------------------
# Here we combine both training and testing dataset for creating dummy variables

all_data <- rbind(house_train,house_test)
glimpse(all_data)
all_data$Address=NULL
sort(table(all_data$Suburb))
sort(table(all_data$Type))
sort(table(all_data$Method))
sort(table(all_data$SellerG))
sort(table(all_data$Postcode))
sort(table(all_data$CouncilArea))
# -------------------------------------------------------------------------------------------------------------
# To create dummy variables
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

col_names<-c("Suburb","Type","Method","SellerG","Postcode","CouncilArea")
for (cat in col_names) {
  all_data=CreateDummies(all_data,cat,100)
}

# ----------------------------------------------------------------------------------------------------------------

# Missing Data Imputation
# Bedroom2 Variable
all_data$Bedroom2[which(is.na(all_data$Bedroom2))]=round(median(all_data$Bedroom2,na.rm =T))

# Bathroom 
all_data$Bathroom[which(is.na(all_data$Bathroom))]=round(median(all_data$Bathroom,na.rm =T))

# Car
all_data$Car[which(is.na(all_data$Car))]=round(median(all_data$Car,na.rm =T))

# Landsize
all_data$Landsize[which(is.na(all_data$Landsize))]=round(median(all_data$Landsize,na.rm =T))

# BuildingArea
all_data$BuildingArea[which(is.na(all_data$BuildingArea))]=round(median(all_data$BuildingArea,na.rm =T))

# YearBuilt
all_data$YearBuilt[which(is.na(all_data$YearBuilt))]=round(median(all_data$YearBuilt,na.rm =T))

# Landsize 
all_data$Landsize[which(all_data$Landsize==0)]=round(median(all_data$Landsize))
# -------------------------------------------------------------------------------------------------------
# all_data<-all_data %>% 
#   mutate(Rooms=as.numeric(scale(all_data$Rooms)),
#          Distance=as.numeric(scale(all_data$Distance)),
#          Bedroom2=as.numeric(scale(all_data$Bedroom2)),
#          Bathroom=as.numeric(scale(all_data$Bathroom)),
#          Car=as.numeric(scale(all_data$Car)),
#          Landsize=as.numeric(scale(all_data$Landsize)),
#          BuildingArea=as.numeric(scale(all_data$BuildingArea)),
#          YearBuilt=as.numeric(scale(all_data$YearBuilt)))
# Seperate Training and Testing dataset

house_train_new<-all_data[which(all_data$ID=="Train"),]

house_train_new<-house_train_new %>% 
  select(-ID)


house_test_new<-all_data[which(all_data$ID=="Test"),]
house_test_new<-house_test_new %>% 
  select(-ID,-Price)

# -------------------------------------------------------------------------------------------------\
# pc<-prcomp(all_data[-c(2,10)],scale. = T,center = T)
# house_test_new<-as.data.frame(pc$x[which(all_data$ID=="Test"),1:75])
# house_train_new<-as.data.frame(pc$x[which(all_data$ID=="Train"),1:75])

# -----------------------------------------------------
# Model
set.seed(1234)
# training data & validation data to check RMSE
train_data<-house_train_new[sample(1:nrow(house_train_new),0.8*nrow(house_train_new)),]
# train_label<- house_train[rownames(train_data),"Price"]
val_data<-house_train_new[-sample(1:nrow(house_train_new),0.8*nrow(house_train_new)),]
# val_label<-house_train[rownames(val_data),"Price"]
f = as.formula(paste(names(train_data[2]),"~",paste(names(train_data[-2]),collapse = "+")))
tic()
set.seed(123)
rf_model<-randomForest(f,
                       data = house_train_new,
                       mtry=50,ntree=450,do.trace=T,maxnodes=100,nodesize=10)
pred<-predict(rf_model,newdata = val_data)
# RMSE
errors<-pred-val_data$Price
errors**2 %>%  mean() %>% sqrt()
toc()
importance(rf_model)
varImpPlot(rf_model)
# Validatation set Prediction
# 
final_rfmodel<-randomForest(f,data = house_train_new,
                            mtry=50,ntree=450,do.trace=T)
importance(final_rfmodel)
varImpPlot(final_rfmodel)

final_pred<-as.data.frame(predict(final_rfmodel,newdata = house_test_new))
names(final_pred)<-"Price"
write.csv(final_pred,"Lokesh_Harad_P1_part2_rf9518.csv",row.names = F)


scatter.smooth(val_data$Price,gbm_pred)
ggplot(val_data,aes(x=gbm_pred,y=val_data$Price))+geom_point()+geom_abline()
# ----------------------------------------------------------------------------------------------------
param=list(n.trees=c(500,1000,1500,2000,2500),
           shrinkage=c(0.1,0.5,0.01,0.05,0.001,0.005),
           n.minobsinnode=c(10),
           interaction.depth=c(1,3,5))

# n.trees=c(500,1000,1500,2000,2500), shrinkage=c(0.1,0.5,0.01,0.05,0.001,0.005),
# n.minobsinnode=c(10,15,20,25), interaction.depth=c(1,3,5,7) as params
subset_paras=function(full_list_para,n=10){
  all_comb=expand.grid(full_list_para)
  s=sample(1:nrow(all_comb),n)
  subset_para=all_comb[s,]
  return(subset_para)
}
num_trials=50
my_params=subset_paras(param,num_trials)
tic()
myerror=99999999

for(i in 1:num_trials){
  print(paste0('starting iteration:',i))
  # uncomment the line above to keep track of progress
  params=my_params[i,]
  k=cvTuning(gbm,Price~.,
             data =house_train_new,
             tuning =params,
             folds = cvFolds(nrow(house_train_new), K=10, type = "random"),
             seed =2
  )
  score.this=k$cv[,2]
  if(score.this<myerror){
    # print(params)
    # uncomment the line above to keep track of progress
    myerror=score.this
    print(myerror)
    # uncomment the line above to keep track of progress
    best_params=params
  }
   print('DONE')
  # uncomment the line above to keep track of progress
}
toc()



# ###############################################################
# gbm with cross validation 

fitControl <- trainControl(
  method = "repeatedcv",
  number = ifelse(grepl("cv",method),10,25),
  ## repeated ten times
  repeats = 10)

set.seed(825)
gbmFit <- train(x=house_train_new[,new_predictor],y=Price, 
                data = house_train_new, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = T, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = param)
