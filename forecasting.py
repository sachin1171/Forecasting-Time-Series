##################### problem 1 ###########################
import pandas as pd
import calendar
air = pd.read_excel("C:/Users/usach/Desktop/Forecasting time series/Airlines Data.xlsx")

dates = pd.date_range(start='1945-01-01', freq='MS',periods=len(air))

air['Months'] = dates.month
air['Year'] = dates.year

air['Months'] = air['Months'].apply(lambda x: calendar.month_abbr[x])

# Pre processing
import numpy as np

air["t"] = np.arange(1,97)

air["t_square"] = air["t"]*air["t"]
air["log_Passengers"] = np.log(air["Passengers"])
air.columns
    
month_dummies = pd.DataFrame(pd.get_dummies(air['Months']))
air1 = pd.concat([air, month_dummies], axis = 1)

# Visualization - Time plot
air1.Passengers.plot()

# Data Partition
Train = air1.head(80)
Test = air1.tail(16)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_Passengers ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Passengers ~ t+t_square', data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('Passengers ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality Linear Trend  ###########

Mul_Add_sea = smf.ols('log_Passengers ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
##################### problem 2 ###########################
import pandas as pd

cock = pd.read_excel("C:/Users/usach/Desktop/Forecasting time series/CocaCola_Sales_Rawdata.xlsx")

month = ['Q1','Q2','Q3','Q4'] 

# Pre processing
import numpy as np
cock["t"] = np.arange(1,43)
cock["t_square"] = cock["t"]*cock["t"]
cock["log_Sales"] = np.log(cock["Sales"])
cock.columns

p = cock["Quarter"][0]
p[0:2]

cock['quarter']= 0

for i in range(42):
    p = cock["Quarter"][i]
    cock['quarter'][i]= p[0:2]
    
Quarter_dummies = pd.DataFrame(pd.get_dummies(cock['quarter']))
cock1 = pd.concat([cock, Quarter_dummies], axis = 1)
# Visualization - Time plot
cock1.Sales.plot()
# Data Partition
Train = cock1.head(35)
Test = cock1.tail(7)
# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))
####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################
Exp = smf.ols('log_Sales ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#################### Quadratic ###############################
Quad = smf.ols('Sales ~ t+t_square', data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad
################### Additive seasonality ########################
add_sea = smf.ols('Sales ~ Q1+Q2+Q3+Q4', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea
################## Multiplicative Seasonality ##################
Mul_sea = smf.ols('log_Sales ~ Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
################## Additive Seasonality Quadratic Trend ############################
add_sea_Quad = smf.ols('Sales ~ t+t_square+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
################## Multiplicative Seasonality Linear Trend  ###########
Mul_Add_sea = smf.ols('log_Sales ~ t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
################## Testing #######################################
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

##################### problem 3 ##########################
import pandas as pd
sale = pd.read_csv("C:/Users/usach/Desktop/Forecasting time series/PlasticSales.csv")

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 

# Pre processing
import numpy as np

sale["t"] = np.arange(1,61)

sale["t_square"] = sale["t"]*sale["t"]
sale["log_Sales"] = np.log(sale["Sales"])
sale.columns
p = sale["Month"][0]
p[0:3]

sale['months']= 0

for i in range(60):
    p = sale["Month"][i]
    sale['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(sale['months']))
sale1 = pd.concat([sale, month_dummies], axis = 1)

# Visualization - Time plot
sale1.Sales.plot()

# Data Partition
Train = sale1.head(50)
Test = sale1.tail(10)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_Sales ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Sales ~ t+t_square', data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Sales ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Sales ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('Sales ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality Linear Trend  ###########

Mul_Add_sea = smf.ols('log_Sales ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
##################### problem 4 ##########################
import pandas as pd
import calendar
solar = pd.read_csv("C:/Users/usach/Desktop/Forecasting time series/solarpower_cumuldaybyday2.csv")

dates = pd.date_range(start='26-10-2011', freq='D',periods=len(solar))

solar['Months'] = dates.month
solar['Months'] = solar['Months'].apply(lambda x: calendar.month_abbr[x])

solar['Day'] = dates.day
solar['Day'] = solar['Day'].apply(lambda x: calendar.month_abbr[x])

solar['Year'] = dates.year

# Pre processing
import numpy as np

solar["t"] = np.arange(1,2559)

solar["t_square"] = solar["t"]*solar["t"]
solar["log_cum_power"] = np.log(solar["cum_power"])
solar.columns    
month_dummies = pd.DataFrame(pd.get_dummies(solar['Months']))
solar1 = pd.concat([solar, month_dummies], axis = 1)

# Visualization - Time plot
solar1.cum_power.plot()

# Data Partition
Train = solar1.head(2400)
Test = solar1.tail(158)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('cum_power ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['cum_power']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_cum_power ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['cum_power']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('cum_power ~ t+t_square', data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['cum_power'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################
add_sea = smf.ols('cum_power ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['cum_power'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_cum_power ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['cum_power']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('cum_power ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['cum_power'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality Linear Trend  ###########

Mul_Add_sea = smf.ols('log_cum_power ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['cum_power'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea
################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
