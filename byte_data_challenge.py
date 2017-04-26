import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

##########first split data, now commit it out######
#data= pd.read_csv('kc_house_train_data.csv')
#train_house, test_house = train_test_split(data, test_size = 0.2)  #split up data
#train_house.to_csv('train_house.csv')
#test_house.to_csv('test_house.csv')


data= pd.read_csv('train_house.csv')
test_data= pd.read_csv('test_house.csv')

#normalize(?)

single_features= list(data.columns.values)
single_features.remove('Unnamed: 0')
single_features.remove('id')
single_features.remove('date')
single_features.remove('price')
print single_features

#for feature in single_features:
#    print feature

small_features='sqft_lot'
medium_features=['sqft_living','bedrooms','zipcode']


prices=data['price']  # target value
small_data=data[small_features]
medium_data=data[medium_features]
large_data=data
del large_data['price']  #delete irrelevant columns
del large_data['id']
del large_data['date']


#Test data and prices
test_price=test_data['price']
small_test=test_data[small_features]
medium_test=test_data[medium_features]
large_test=test_data
del large_test['price']  #delete irrelevant columns
del large_test['id']
del large_test['date']











###if want to see features individually
##for feature in single_features:
##    print feature
##    small_data=data[feature]
##    linear = LinearRegression()
##    linear.fit(small_data.to_frame(), prices.to_frame())  
##    print linear.coef_
##    X=linear.predict(small_test.to_frame())  # this is a numpy.ndarray
##    Y=pd.DataFrame(X)
##    print np.mean(((test_price - Y)[0])**2)





#Linear Regression for 1 feature
linear = LinearRegression()
linear.fit(small_data.to_frame(), prices.to_frame())  
print linear.coef_
X=linear.predict(small_test.to_frame())  # this is a numpy.ndarray
Y=pd.DataFrame(X)
print np.mean(((test_price - Y)[0])**2)  # this is producing a 3477*3477 - y?



#improve visualizations
plt.scatter(small_test, test_price,  color='black')
plt.plot(small_test, linear.predict(small_test.to_frame()), color='blue',
         linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
 



#Linear Regression for 3 chosen features
linear = LinearRegression()

linear.fit(medium_data, prices.to_frame())  
print linear.coef_
X=linear.predict(medium_test)  # this is a numpy.ndarray
Y=pd.DataFrame(X)
print np.mean(((test_price - Y)[0])**2)



#Linear Regression for all 19 features
linear = LinearRegression()

linear.fit(large_data, prices.to_frame())  
print linear.coef_
X=linear.predict(large_test)  # this is a numpy.ndarray
Y=pd.DataFrame(X)
print np.mean(((test_price - Y)[0])**2)


#SECOND PART




