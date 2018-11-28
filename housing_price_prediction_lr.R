# -----------------------------------------------##########-------------------------------------------------

## Name of Project : Predict a House Price using Predictive Model(Linear Regression) 
## Project Member  : Lokesh Daulat Harad
## Date : 12 April 2018
## Edvancer Edvantures

#------------------------------------------------##########--------------------------------------------------

# Load packages 
library(psych)
library(dplyr)
library(tidyr)
library(ggplot2)
library(mice)
#----------------------------------------------------------------------------------------------------------------

# Load a Training dataset 
house_train <- read.csv("D:\\edvancer\\business_analytics\\Projects\\1_Real_Estate\\housing_train.csv",stringsAsFactors = F)
View(house_train)


#  Here we create ID as feature to identify Training dataset 
house_train <- house_train %>% 
  mutate(ID = "Train")
# ---------------------------------------------------------------

# Load a Testing dataset
house_test <- read.csv("D:\\edvancer\\business_analytics\\Projects\\1_Real_Estate\\housing_test.csv",stringsAsFactors = F)
View(house_test)

# We add two features or variables ID and Price 
# because below we create all_data dataset for assinging new features 
#  or dummy variables from old variales like Suburb, Type, Method,etc.

house_test <- house_test %>% 
  mutate(Price = NA,
         ID    = "Test")

# ----------------------------------------------------------------
# Here we combine both training and testing dataset for creating dummy variables

all_data <- rbind(house_train,house_test)
View(all_data)

# -------------------------------------------------------------------------------------------------------------

# 1. DUMMY VARIABLES FOR SUBURB VARIABLE

sort(table(all_data$Suburb))

# here After sorting table, set cutoff = 150

all_data <- all_data %>% 
  mutate(sub_reservoir    = as.numeric(Suburb=="Reservoir"),
         sub_bentleigh_E  = as.numeric(Suburb=="Bentleigh East"),
         sub_richmond     = as.numeric(Suburb=="Richmond"),
         sub_preston      = as.numeric(Suburb=="Preston"),
         sub_st_kilda     = as.numeric(Suburb=="St Kilda"),
         sub_south_yarra  = as.numeric(Suburb=="South Yarra"),
         sub_brunswick    = as.numeric(Suburb=="Brunswick"),
         sub_essndon      = as.numeric(Suburb=="Essendon"),
         sub_glen_iris    = as.numeric(Suburb=="Glen Iris"),
         sub_glenroy      = as.numeric(Suburb=="Glenroy"),
         sub_brighton     = as.numeric(Suburb=="Brighton")) %>% 
  select(-Suburb)
# -----------------------------------------------------------

# Address variable is discarded from this dataset

all_data$Address=NULL

# ----------------------------------------------------------------------------------------------------------

# 2. DUMMY VARIABLE FOR TYPE 

# there are 3 categories in Type variable so we take all categories as 
# dummy vbariable using model.matrix() function

sort(table(all_data$Type))

# to create dummy variable we take (n-1) dummy variable from categories

Type_Model  <- data.frame(model.matrix(all_data$Rooms ~ all_data$Type-1,all_data))

all_data <- all_data %>% 
  mutate(type_h = Type_Model$all_data.Typeh,
         type_t = Type_Model$all_data.Typet) %>% 
  select(-Type)

# -----------------------------------------------------------------------------------------------------------

# 3. DUMMY VARAIBLES FOR METHOD VARIABLE

# there are 5 categories in Method variable such as SA, VB, SP, PI, S
# we take only 4 categories except SA, because no. of obs are  rare for this Category

sort(table(all_data$Method))

Method_Model <- data.frame(model.matrix(all_data$Rooms ~ all_data$Method-1,all_data))

all_data  <- all_data %>% 
  mutate(s_method  = Method_Model$all_data.MethodS,
         pi_method = Method_Model$all_data.MethodPI,
         sp_method = Method_Model$all_data.MethodSP,
         vb_method = Method_Model$all_data.MethodVB) %>% 
  select(-Method)

# ------------------------------------------------------------------------------------------------------------

# 4. Dummy Variables for SellerG variable

# We take 5 categories as dummy variables for SellerG Variable
# setting cutoff = 500

sort(table(all_data$SellerG))

SellerG_Model <- data.frame(model.matrix(all_data$Rooms ~ all_data$SellerG-1,all_data))
all_data   <- all_data %>% 
  mutate(sel_nelson     = SellerG_Model$all_data.SellerGNelson,
         sel_jellis     = SellerG_Model$all_data.SellerGJellis,
         sel_hockstuart = SellerG_Model$all_data.SellerGhockingstuart,
         sel_barry      = SellerG_Model$all_data.SellerGBarry,
         sel_marshal    = SellerG_Model$all_data.SellerGMarshall) %>% 
         # sel_buxton     = SellerG_Model$all_data.SellerGBuxton,
         # sel_ray        = SellerG_Model$all_data.SellerGRay,
         # sel_biggin     = SellerG_Model$all_data.SellerGBiggin,
         # sel_brad       = SellerG_Model$all_data.SellerGBrad)%>% 
  select(-SellerG)

# -----------------------------------------------------------------------------------------------------------------

# 5. Dummy Variables for PostCode
#

class(all_data$Postcode)

# class of postcode is numerical and as per given as catogorical,
# so we change from numerical to character
# setting cutoff = 150 for dummy variables

all_data <- all_data %>%
  mutate(postcode = as.character(Postcode)) %>%
  select(-Postcode)

sort(table(all_data$postcode))

Post_Model  <- data.frame(model.matrix(all_data$Rooms ~ all_data$postcode-1,all_data))
all_data <- all_data %>% 
  mutate(code_3073 = Post_Model$all_data.postcode3073,
         code_3020 = Post_Model$all_data.postcode3020,
         code_3165 = Post_Model$all_data.postcode3165,
         code_3046 = Post_Model$all_data.postcode3046,
         code_3121 = Post_Model$all_data.postcode3121,
         code_3032 = Post_Model$all_data.postcode3032,
         code_3163 = Post_Model$all_data.postcode3163,
         code_3058 = Post_Model$all_data.postcode3058,
         code_3040 = Post_Model$all_data.postcode3040,
         code_3204 = Post_Model$all_data.postcode3204,
         code_3072 = Post_Model$all_data.postcode3072,
         code_3182 = Post_Model$all_data.postcode3182,
         code_3141 = Post_Model$all_data.postcode3141,
         code_3012 = Post_Model$all_data.postcode3012,
         code_3056 = Post_Model$all_data.postcode3056,
         code_3146 = Post_Model$all_data.postcode3146,
         code_3084 = Post_Model$all_data.postcode3084,
         code_3186 = Post_Model$all_data.postcode3186) %>% 
  select(-postcode)

# --------------------------------------------------------------------------------------------------------------------

# 6. Dummy variables for CouncilArea

sort(table(all_data$CouncilArea))

# we take mean of price after grouping CouncilArea 
# set ranges as per below like 400000-500000 : 4_5_100k

sort(round(tapply(all_data$Price,all_data$CouncilArea,mean,na.rm=T)))

all_data      <- all_data %>% 
  mutate(ca_4_5_100k    = as.numeric(CouncilArea %in% c("Hume")),
         ca_6_7_100k    = as.numeric(CouncilArea %in% c("Brimbank")),
         ca_7_8_100k    = as.numeric(CouncilArea %in% c("Maribyrnong","Moreland")),
         ca_8_9_100k    = as.numeric(CouncilArea %in% c("Darebin")),
         ca_9_10_100k   = as.numeric(CouncilArea %in% c("Banyule","Melbourne",
                                                        "Kingston","Hobsons Bay")),
         ca_10_11_100k  = as.numeric(CouncilArea %in% c("Monash","",
                                                        "Glen Eira","Port Phillip")),
         ca_11_12_100k  = as.numeric(CouncilArea %in% c("Yarra","Manningham")),
         ca_13_14_100k  = as.numeric(CouncilArea %in% c("Whitehorse","Stonnington")),
         ca_16_17_100k  = as.numeric(CouncilArea %in% c("Boroondara","Bayside"))) %>% 
  select(-CouncilArea)

# ----------------------------------------------------------------------------------------------------------------

# Missing Data Imputation
# Bedroom Variable
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
# -------------------------------------------------------------------------------------------------------

# Seperate Training and Testing dataset

house_train_new<-all_data[which(all_data$ID=="Train"),]
house_train_new<-house_train_new %>% 
  select(-ID)

house_test_new<-all_data[which(all_data$ID=="Test"),]
house_test_new<-house_test_new %>% 
  select(-ID,-Price)

# -------------------------------------------------------------------------------------------------

# Model
set.seed(1234)
# training data & validation data to check RMSE
train_data<-house_train_new[sample(1:nrow(house_train_new),0.8*nrow(house_train_new)),]
val_data<-house_train_new[-sample(1:nrow(house_train_new),0.8*nrow(house_train_new)),]

# --------------------------------------------------------------------------------
# Random Forest ALgorithm

# ---------------------------------------------------------------------------------------------------------
 model<-lm(log(train_data$Price)+1~.,data = train_data)
summary(model)
model<-step(model)

# 
# 
# # Model after AIC
model<-lm(formula =log(train_data$Price) + 1 ~ Rooms + Distance + Bedroom2 + Bathroom + 
            Car + Landsize + BuildingArea + YearBuilt + sub_reservoir + 
            sub_bentleigh_E + sub_preston + sub_st_kilda + sub_glenroy + 
            sub_brighton + type_h + type_t + pi_method + sp_method + 
            vb_method + sel_nelson + sel_jellis + sel_hockstuart + sel_barry + 
            sel_marshal + code_3020 + code_3046 + code_3121 + code_3032 + 
            code_3163 + code_3058 + code_3040 + code_3204 + code_3012 + 
            ca_4_5_100k + ca_7_8_100k + ca_8_9_100k + ca_9_10_100k + 
            ca_10_11_100k + ca_11_12_100k + ca_13_14_100k + ca_16_17_100k,data = train_data)
summary(model)
# 
# # ------------------------------------------------------------------------------------
# 
# # Prediction on validation data
# 
pred = exp(predict(model,newdata = val_data))-1
errors<-pred-val_data$Price
errors**2 %>%  mean() %>% sqrt()

# # ----------------------------------------------------------------
# 
# # Here final model is for prdiction on test data set
# 
final_model<-lm(log(house_train_new$Price)+1~.,data=house_train_new)
summary(final_model)

final_model<-step(final_model)
final_model<-lm(formula = log(house_train_new$Price) + 1 ~ Rooms + Distance + Bedroom2 + 
                  Bathroom + Car + Landsize + BuildingArea + YearBuilt + sub_reservoir + 
                  sub_bentleigh_E + sub_preston + sub_brunswick + sub_glenroy + 
                  sub_brighton + type_h + type_t + pi_method + sp_method + 
                  vb_method + sel_nelson + sel_jellis + sel_hockstuart + sel_barry + 
                  sel_marshal + code_3020 + code_3046 + code_3121 + code_3032 + 
                  code_3163 + code_3058 + code_3040 + code_3204 + code_3012 + 
                  ca_4_5_100k + ca_7_8_100k + ca_8_9_100k + ca_9_10_100k + 
                  ca_10_11_100k + ca_11_12_100k + ca_13_14_100k + ca_16_17_100k,
                data=house_train_new)
summary(final_model)

# # ------------------------------------------------------------------------
# 
# # Prediction on test dataset
# 
test_pred<-exp(predict(final_model,newdata = house_test_new))-1
# 
# # CSV file of predicted Price 
# 
write.csv(test_pred,"Lokesh_Harad_P1_part2_1_aftr_log.csv",row.names = F)
# 
# # ---------------------------------------------------------------
# 
plot(final_model,1)# Residuals
plot(final_model,2)# Normal Q-Q
plot(final_model,3)# Scale Location
plot(final_model,4)# Cook's Distance
# 
# # --------------------------------------------------------------------------------------------------------