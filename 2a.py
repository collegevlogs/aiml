import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv('student.csv')
print(data.shape)
data.head()
math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data['Math'], data['Reading'], data['Writing'], color='red')
plt.title('Math vs Reading vs Writing')
plt.show()
m=len(math)
x0=np.ones(m)
X=np.array([x0,math,read]).T
Y = np.array(write)
B = np.array([0, 0, 0])    
alpha = 0.0001          
def cost_function(X,Y,B):
    m=len(Y)
    j=np.sum((X.dot(B)-Y)**2)/2*m
    return j
initial_cost = cost_function(X, Y, B)
print("Initial Cost:", initial_cost)
def gradient_descent(X,Y,B,alpha,iterations):
    cost_history=[0]*iterations
    m=len(Y)
    for iteration in range(iterations):
            h=X.dot(B)
            loss=h-Y
            gardient=X.T.dot(loss)/m
            B=B-alpha*gardient
            cost=cost_function(X,Y,B)
            cost_history[iteration]=cost
    return B,cost_history
newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)
print("New Coefficients (b0, b1, b2):", newB)
Y_pred = X.dot(newB)
mean_y = np.mean(Y)
ss_tot = np.sum((Y - mean_y)**2)
ss_res = np.sum((Y - Y_pred)**2)
r2 = 1 - (ss_res / ss_tot)
print("R² Score:", r2)

