import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv("housing_prices.csv")
print(data.head())
print(data.shape)
X=data.iloc[:,:3].values
Y=data.iloc[:,4].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
model = LinearRegression()
model.fit(X_train, Y_train)
print("Intercept (b₀):", model.intercept_)
print("Coefficients (b₁, b₂, b₃):", model.coef_)
r2_train = model.score(X_train, Y_train)
r2_test = model.score(X_test, Y_test)
print("R² Score (Training):", r2_train)
print("R² Score (Testing):", r2_test)
