import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("ch1ex1.csv")
print(data.head())
points = data.values
model = KMeans(3)
model.fit(points)
labels = model.predict(points) 
xs = points[:, 0]   
ys = points[:, 1]
plt.scatter(xs, ys, c=labels, cmap='rainbow')
plt.title("K-Means Clustering (K=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
centroids = model.cluster_centers_
print("\nCluster Centers:\n", centroids)
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]
plt.scatter(xs, ys, c=labels, cmap='rainbow')
plt.scatter(centroids_x, centroids_y, marker='X', s=200, color='black')
plt.title("K-Means Clustering with Centroids (K=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
