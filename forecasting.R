#####################problem 1 #######################
library(forecast)
library(fpp)
library(smooth)
library(readxl)

Airlines<-read_excel(file.choose()) # read the Airlines data
View(Airlines) # Seasonality 12 months 
windows()
plot(Airlines$Passengers,type="o")

# So creating 12 dummy variables 

X<- data.frame(outer(rep(month.abb,length = 96), month.abb,"==") + 0 )# Creating dummies for 12 months
# View(X)

colnames(X)<-month.abb # Assigning month names 
# View(X)
AirlinesData<-cbind(Airlines,X)
View(AirlinesData)
colnames(AirlinesData)


AirlinesData["t"]<- 1:96
View(AirlinesData)
AirlinesData["log_Passenger"]<-log(AirlinesData["Passengers"])
AirlinesData["t_square"]<-AirlinesData["t"]*AirlinesData["t"]
attach(AirlinesData)

train<-AirlinesData[1:84,]

test<-AirlinesData[85:96,]

########################### LINEAR MODEL #############################

linear_model<-lm(Passengers~t,data=train)
summary(linear_model)

linear_pred<-data.frame(predict(linear_model,interval='predict',newdata =test))
View(linear_pred)
rmse_linear<-sqrt(mean((test$Passengers-linear_pred$fit)^2,na.rm = T))
rmse_linear # 53.19924
######################### Exponential #################################

expo_model<-lm(log_Passenger~t,data=train)
summary(expo_model)

expo_pred<-data.frame(predict(expo_model,interval='predict',newdata=test))
rmse_expo<-sqrt(mean((test$Passengers-exp(expo_pred$fit))^2,na.rm = T))
rmse_expo # 46.05736  and Adjusted R2 - 82.18 %
######################### Quadratic ####################################
Quad_model<-lm(Passengers~t+t_square,data=train)
summary(Quad_model)

Quad_pred<-data.frame(predict(Quad_model,interval='predict',newdata=test))
rmse_Quad<-sqrt(mean((test$Passengers-Quad_pred$fit)^2,na.rm=T))
rmse_Quad # 48.05189 and Adjusted R2 - 79.12%


######################### Additive Seasonality #########################

sea_add_model<-lm(Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data=train)
summary(sea_add_model)

sea_add_pred<-data.frame(predict(sea_add_model,newdata=test,interval='predict'))
rmse_sea_add<-sqrt(mean((test$Passengers-sea_add_pred$fit)^2,na.rm = T))
rmse_sea_add # 132.8198


######################## Additive Seasonality with Linear #################

Add_sea_Linear_model<-lm(Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data=train)
summary(Add_sea_Linear_model)
Add_sea_Linear_pred<-data.frame(predict(Add_sea_Linear_model,newdata=test,interval='predict'))
rmse_sea_add<-sqrt(mean((test$Passengers-Add_sea_Linear_pred$fit)^2,na.rm = T))
rmse_sea_add 


######################## Additive Seasonality with Quadratic #################

Add_sea_Quad_model<-lm(Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data=train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred<-data.frame(predict(Add_sea_Quad_model,newdata=test,interval='predict'))
rmse_sea_add<-sqrt(mean((test$Passengers-Add_sea_Quad_pred$fit)^2,na.rm = T))
rmse_sea_add 

######################## Multiplicative Seasonality #########################

multi_sea_model<-lm(log_Passenger~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data = train)
summary(multi_sea_model)
multi_sea_pred<-data.frame(predict(multi_sea_model,newdata=test,interval='predict'))
rmse_multi_sea<-sqrt(mean((test$Passengers-exp(multi_sea_pred$fit))^2,na.rm = T))
rmse_multi_sea 
######################## Multiplicative Seasonality Linear trend ##########################

multi_add_sea_model<-lm(log_Passenger~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data = train)
summary(multi_add_sea_model) 

multi_add_sea_pred<-data.frame(predict(multi_add_sea_model,newdata=test,interval='predict'))
rmse_multi_add_sea<-sqrt(mean((test$Passengers-exp(multi_add_sea_pred$fit))^2,na.rm = T))
rmse_multi_add_sea # 10.51917 and Adjusted R2 - 97.23%

# Preparing table on model and it's RMSE values 

table_rmse<-data.frame(c("rmse_linear","rmse_expo","rmse_Quad","rmse_sea_add","rmse_multi_add_sea"),c(rmse_linear,rmse_expo,rmse_Quad,rmse_sea_add,rmse_multi_add_sea))
View(table_rmse)
colnames(table_rmse)<-c("model","RMSE")
View(table_rmse)

# Multiplicative Seasonality Linear trend  has least RMSE value

new_model<-lm(log_Passenger~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data = AirlinesData)
new_model_pred<-data.frame(predict(new_model,newdata=AirlinesData,interval='predict'))
new_model_fin <- exp(new_model$fitted.values)

View(new_model_fin)

pred_res<- predict(arima(log_Passenger,order=c(1,0,0)),n.ahead = 12)
Month <- as.data.frame(Airlines$Month)

Final <- as.data.frame(cbind(Month,AirlinesData$Passengers,new_model_fin))
colnames(Final) <-c("Month","Passengers","New_Pred_Value")
Final <- as.data.frame(Final)
View(Final)

#####################problem  2##########################
library(readxl)
coke <- read_xlsx(file.choose()) # read the cocacola data
View(coke) # Seasonality 12 months

# Pre Processing
# input t
coke["t"] <- c(1:42)
View(coke)

coke["t_square"] <- coke["t"] * coke["t"]
coke["log_sales"] <- log(coke["Sales"])
#plot ACf
plot(coke$Sales,type="o")

# So creating 4 dummy variables 
Q1 <-  ifelse(grepl("Q1",coke$Quarter),'1','0')
Q2 <-  ifelse(grepl("Q2",coke$Quarter),'1','0')
Q3 <-  ifelse(grepl("Q3",coke$Quarter),'1','0')
Q4 <-  ifelse(grepl("Q4",coke$Quarter),'1','0')

coke1<-cbind(coke,Q1,Q2,Q3,Q4)
View(coke1)
colnames(coke1)
## Pre-processing completed

attach(coke1)

# partitioning
train <- coke1[1:35, ]
test <- coke1[36:42, ]

########################### LINEAR MODEL #############################

linear_model <- lm(Sales ~ t, data = train)
summary(linear_model)
linear_pred <- data.frame(predict(linear_model, interval = 'predict', newdata = test))
rmse_linear <- sqrt(mean((test$Sales - linear_pred$fit)^2, na.rm = T))
rmse_linear

######################### Exponential ############################

expo_model <- lm(log_sales ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval = 'predict', newdata = test))
rmse_expo <- sqrt(mean((test$Sales - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo

######################### Quadratic ###############################

Quad_model <- lm(Sales ~ t + t_square, data = train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval = 'predict', newdata = test))
rmse_Quad <- sqrt(mean((test$Sales-Quad_pred$fit)^2, na.rm = T))
rmse_Quad

######################### Additive Seasonality #########################

sea_add_model <- lm(Sales ~ Q1 +Q2+ Q3 + Q4, data = train)
summary(sea_add_model)
sea_add_pred <- data.frame(predict(sea_add_model, newdata = test, interval = 'predict'))
rmse_sea_add <- sqrt(mean((test$Sales - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add
######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_sales ~ Q1 +Q2+ Q3 + Q4, data = train)
summary(multi_sea_model)
multi_sea_pred <- data.frame(predict(multi_sea_model, newdata = test, interval = 'predict'))
rmse_multi_sea <- sqrt(mean((test$Sales - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea

################### Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model <- lm(Sales ~ t + t_square + Q1 +Q2+ Q3 + Q4, data = train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model, interval = 'predict', newdata = test))
rmse_Add_sea_Quad <- sqrt(mean((test$Sales - Add_sea_Quad_pred$fit)^2, na.rm=T))
rmse_Add_sea_Quad

# Preparing table on model and it's RMSE values 
table_rmse <- data.frame(c("rmse_linear", "rmse_expo", "rmse_Quad", "rmse_sea_add", "rmse_Add_sea_Quad", "rmse_multi_sea"), c(rmse_linear, rmse_expo, rmse_Quad, rmse_sea_add, rmse_Add_sea_Quad, rmse_multi_sea))
colnames(table_rmse) <- c("model", "RMSE")
View(table_rmse)

################### problem 3 #######################
library(readr)
plastic <- read.csv(file.choose()) 
View(plastic) # Seasonality 12 months

# Pre Processing
# input t
plastic["t"] <- c(1:60)
View(plastic)

plastic["t_square"] <- plastic["t"] * plastic["t"]
plastic["log_sales"] <- log(plastic["Sales"])

plot(plastic$Sales,type="o")
# So creating 12 dummy variables
X <- data.frame(outer(rep(month.abb,length = 60), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X) <- month.abb # Assigning month names
View(X)

plasticsale <- cbind(plastic, X)
colnames(plasticsale)

View(plasticsale)
## Pre-processing completed

attach(plasticsale)

# partitioning
train <- plasticsale[1:50, ]
test <- plasticsale[50:60, ]

########################### LINEAR MODEL #############################

linear_model <- lm(Sales ~ t, data = train)
summary(linear_model)

linear_pred <- data.frame(predict(linear_model, interval = 'predict', newdata = test))

rmse_linear <- sqrt(mean((test$Sales - linear_pred$fit)^2, na.rm = T))
rmse_linear

######################### Exponential ############################

expo_model <- lm(log_sales ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval = 'predict', newdata = test))
rmse_expo <- sqrt(mean((test$Sales - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo

######################### Quadratic ###############################

Quad_model <- lm(Sales ~ t + t_square, data = train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval = 'predict', newdata = test))
rmse_Quad <- sqrt(mean((test$Sales-Quad_pred$fit)^2, na.rm = T))
rmse_Quad

######################### Additive Seasonality #########################

sea_add_model <- lm(Sales ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec, data = train)
summary(sea_add_model)
sea_add_pred <- data.frame(predict(sea_add_model, newdata = test, interval = 'predict'))
rmse_sea_add <- sqrt(mean((test$Sales - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add


######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_sales ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(multi_sea_model)
multi_sea_pred <- data.frame(predict(multi_sea_model, newdata = test, interval = 'predict'))
rmse_multi_sea <- sqrt(mean((test$Sales - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea

################### Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model <- lm(Sales ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model, interval = 'predict', newdata = test))
rmse_Add_sea_Quad <- sqrt(mean((test$Sales - Add_sea_Quad_pred$fit)^2, na.rm=T))
rmse_Add_sea_Quad

# Preparing table on model and it's RMSE values 
table_rmse <- data.frame(c("rmse_linear", "rmse_expo", "rmse_Quad", "rmse_sea_add", "rmse_Add_sea_Quad", "rmse_multi_sea"), c(rmse_linear, rmse_expo, rmse_Quad, rmse_sea_add, rmse_Add_sea_Quad, rmse_multi_sea))
colnames(table_rmse) <- c("model", "RMSE")
View(table_rmse)

# Additive seasonality with Quadratic Trend has least RMSE value.
###################problem 4 #########################
library(forecast)
library(fpp)
library(smooth)

library(readr)

solar<-read_csv(file.choose()) 
View(solar) # Seasonality 12 months 
windows()
plot(solar$cum_power,type="o")

solar["t"]<- 1:2558
View(solar)
solar["log_cum_power"]<-log(solar["cum_power"])
solar["t_square"]<-solar["t"]*solar["t"]
attach(solar)
#partition
train<-solar[1:2058,]

test<-solar[2059:2558,]

########################### LINEAR MODEL #############################

linear_model<-lm(cum_power~t,data=train)
summary(linear_model)

linear_pred<-data.frame(predict(linear_model,interval='predict',newdata =test))
View(linear_pred)
rmse_linear<-sqrt(mean((test$cum_power-linear_pred$fit)^2,na.rm = T))
rmse_linear 
######################### Exponential #################################
expo_model<-lm(log_cum_power~t,data=train)
summary(expo_model)

expo_pred<-data.frame(predict(expo_model,interval='predict',newdata=test))
rmse_expo<-sqrt(mean((test$cum_power-exp(expo_pred$fit))^2,na.rm = T))
rmse_expo

######################### Quadratic ####################################
Quad_model<-lm(cum_power~t+t_square,data=train)
summary(Quad_model)

Quad_pred<-data.frame(predict(Quad_model,interval='predict',newdata=test))
rmse_Quad<-sqrt(mean((test$cum_power-Quad_pred$fit)^2,na.rm=T))
rmse_Quad # 48.05189 and Adjusted R2 - 79.12%


# Preparing table on model and it's RMSE values 

table_rmse<-data.frame(c("rmse_linear","rmse_expo","rmse_Quad"),c(rmse_linear,rmse_expo,rmse_Quad))
View(table_rmse)
colnames(table_rmse)<-c("model","RMSE")
View(table_rmse)
