import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('headbrain.csv')
print(data.shape) 
data.head()
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values
mean_x=np.mean(X)
mean_y=np.mean(Y)
n=len(X)
numer=0
denom=0
for i in range(n):
    numer+=(X[i]-mean_x)*(Y[i]-mean_y)
    denom+=(X[i]-mean_x)**2
b1=numer/denom
b0=mean_y-(b1*mean_x)
print("b1=",b1)
print("b0=",b0)
x = np.linspace(np.min(X)-100, np.max(X)+100, 1000)
y=b0+b1*x
plt.plot(x,y,c='green',label='Regression Line')
plt.scatter(X,Y,c='blue',label='scatter plot')
plt.xlabel('Head Size (cm³)')
plt.ylabel('Brain Weight (grams)')
plt.legend()
plt.show()
ss_tot = 0
ss_res = 0
for i in range(n):
    y_pred = b0 + b1*X[i]
    ss_tot += (Y[i] - mean_y)**2
    ss_res += (Y[i] - y_pred)**2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score:", r2)
