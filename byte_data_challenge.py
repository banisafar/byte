import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

np.random.seed(1)  # to get the same split everytime


all_data= pd.read_csv('kc_house_train_data.csv')
data, test_data = train_test_split(all_data, test_size = 0.2)  #split up data

#################### definitions ################

def make_a_feature_list(data,feat):
    X = data[feat]
    Y = sm.add_constant(X)  # add a constant
    return Y

def make_a_price_list(data,feat):
    return data[feat]

def solve_MSE(test,Y):
    return np.mean(((test - Y))**2)

def OLS_fit_predict(Y,X,X_test):   
    results = sm.OLS(Y, X).fit()
    y_pred = results.predict(X_test)
    return y_pred

#################### FIRST PART ##########################################
all_features= list(data.columns.values)
all_features.remove('id')  # irrelevant features - remove
all_features.remove('date')
all_features.remove('price')


prices=data['price']  # target value
test_price=test_data['price']


table=pd.DataFrame(index=all_features, columns=['MSE']) # empty dataframe

for feature in all_features:
    single_feature = make_a_feature_list(data,feature)
    single_feature_test=make_a_feature_list(test_data,feature)
    y_pred = OLS_fit_predict(prices,single_feature,single_feature_test)
    MSEE= solve_MSE(test_price,y_pred)
    table.set_value(feature,'MSE', MSEE)


print table[table['MSE'] == min(table['MSE'])] # this is the min MSE
print table[table['MSE'] == max(table['MSE'])]  # this is the max MSE

# we see that longitude has the smalleset MSE - a better analysis would be to look at correlation.  
#can see multiple analysis when look at summary 

###################### SECOND PART #######################################
small_features = 'sqft_living'
medium_features = ['sqft_living','bedrooms','bathrooms']
large_features = all_features  # id/date/price removed

small_data = make_a_feature_list(data,small_features)  # data
medium_data = make_a_feature_list(data,medium_features)
large_data = make_a_feature_list(data,large_features)

small_test = make_a_feature_list(test_data,small_features)  # test data
medium_test = make_a_feature_list(test_data,medium_features)
large_test = make_a_feature_list(test_data,large_features)

#Linear Regression for 1 feature
y_pred = OLS_fit_predict(prices,small_data,small_test)
print(len(y_pred))
MSEE= solve_MSE(test_price,y_pred)
print MSEE

#Linear Regression for 3 chosen features
y_pred = OLS_fit_predict(prices,medium_data,medium_test)
MSEE= solve_MSE(test_price,y_pred)
print MSEE

#Linear Regression for all 19 features
y_pred = OLS_fit_predict(prices,large_data,large_test)
MSEE= solve_MSE(test_price,y_pred)
print MSEE

#######################  THIRD PART #####################################
zipcode_data = data.groupby('zipcode').aggregate(np.mean) # averages the means of all
print zipcode_data[zipcode_data['price'] == max(zipcode_data['price'])]['price'] # see that certain areas are pricer than others
print zipcode_data[zipcode_data['price'] == min(zipcode_data['price'])]['price']
print zipcode_data[zipcode_data['price'] > 1000000]['price']



