import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
df = pd.read_csv('seeds-less-rows.csv')
varieties = list(df.pop('grain_variety'))
samples = df.values
mergings=linkage(samples,method='complete')
dendrogram(mergings,labels=varieties,leaf_rotation=90,leaf_font_size=6)
plt.title("Hierarchical Clustering Dendrogram for Seed Varieties")
plt.xlabel("Grain Samples")
plt.ylabel("Distance")
plt.show()
labels=fcluster(mergings,6,criterion='distance')
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
ct = pd.crosstab(df['labels'], df['varieties'])
ct
