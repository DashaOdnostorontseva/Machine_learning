
import pandas as pd
import numpy as np

import datetime as dt
import matplotlib.pyplot as plt

from matplotlib import dates

from sklearn.linear_model import LinearRegression


from sklearn.preprocessing import PolynomialFeatures 


dataset = pd.read_csv("DataBase.csv", sep=',', decimal=".",   header='infer')


print(dataset.shape )

print(dataset.head())    
print(dataset.columns.values)


dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Date'] = dataset['Date'].map(dt.datetime.toordinal)

dataset=dataset.sort_values('Date')
print(dataset.head())

X = dataset.iloc[:,0].values.reshape(-1,1)
y = dataset.iloc[:,1].values




lin_reg = LinearRegression() 
lin_reg.fit(X,y)



#Вернуть коэффициент детерминации R^{2} предсказания.
r_sq = lin_reg.score(X, y)
print('coefficient of determination R^2 :', r_sq)
# The coefficients
print('Coefficients:')
a = lin_reg.intercept_
b = lin_reg.coef_
print('a:', lin_reg.intercept_)
print('b:', lin_reg.coef_)


#
#
#
#
#


y_lin_regression = a + b*X

# Визуализация результатов линейной регрессионной модели

plt.scatter(X, y, color="red") 
plt.plot(X, y, color="blue")

plt.plot(X, y_lin_regression, color="green")

plt.title("Линейная регрессия") 
plt.xlabel('X') 
plt.ylabel('y') 
plt.show() 


# Подгонка полиномиальной регрессионной модели 


poly_reg = PolynomialFeatures(degree=3) 
X_poly = poly_reg.fit_transform(X) 
print(X_poly)
# prints the X_poly 




#Визуализация полиномиальной регрессионной модели


from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree=3) 
X_poly = poly_reg.fit_transform(X) 
lin_reg2 = LinearRegression() 
lin_reg2.fit(X_poly,y) 
 
X_grid = np.arange(min(X),max(X),0.1) 
X_grid = X_grid.reshape(len(X_grid),1)  
plt.scatter(X,y, color='red')  
 
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')  
 
plt.title("Truth or Bluff (Polynomial)") 
plt.xlabel('X') 
plt.ylabel('y') 

plt.show() 



#
#




#Прогнозирование результата 
#lin_reg.predict() 
#lin_reg2.predict()


print("")

print("END.")

#

