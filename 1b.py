import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv("housing_prices_SLR.csv")
print(df.shape)
df.head()
x = df[['AREA']].values     
y = df['PRICE'].values  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
print(x_train.shape)
x_test.shape
model = LinearRegression()
model.fit(x_train, y_train)
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("R² (Training):", r2_train)
print("R² (Testing):", r2_test)
plt.scatter(x_train, y_train, color='red', label='Training Data')
plt.scatter(x_test, y_test, color='blue', label='Testing Data')
plt.plot(x_train, model.predict(x_train), color='yellow', label='Fit Line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.show()
