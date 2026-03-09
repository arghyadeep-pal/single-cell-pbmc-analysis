# single-cell-pbmc-analysis
Analysis of PBMC 3k dataset using Scanpy and scVI for potential batch correction

# Clustering after pre-processing and PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Generate Synthetic "Cell vs Gene" Data
# 1000 cells are the rows and 2000 genes are the columns (2000 dimensions)
# Imagine 3 cell types: Neurons, Astrocytes, and Microglia
np.random.seed(42)
n_cells = 1000
n_genes = 2000

# Create baseline noise
data = np.random.poisson(lam=1, size=(n_cells, n_genes)).astype(float)

data

data.shape

# Inject "Marker Genes" for 3 specific clusters

# We try to manipulate 150 genes for this example, so as to make them marker genes
# "marker genes" are the ones which actively influence how the cell functions, rest all are "housekeeping genes" which do not affect 
# the cell's functions. Rather they contribute to background noise in the data

# And we influence these 150 marker genes for all the 1000 cells to test out how the clustering will challenge itself and perform

data[0:333, 0:50] += 5    # Cluster 1: High expression in first 50 genes
data[333:666, 50:100] += 8  # Cluster 2: High expression in next 50 genes
data[666:, 100:150] += 4   # Cluster 3: High expression in third 50 genes

data

# 2. Pre-processing: Standardize the data
# Gene expression scales vary wildly, so we normalize.
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

data_scaled

data_scaled.shape

# 3. Dimensionality Reduction (PCA)
# Clustering 2,000 dimensions directly is computationally heavy and noisy.
# We reduce it to the top 50 "Principal Components".

# Using too few PCs doen't give the complete picture


pca = PCA(n_components=50)
pca_results = pca.fit_transform(data_scaled)
pca_results

pca_results.shape

# 4. Clustering (K-Means)
# We assume we are looking for 3 cell types.

# what happens in clustering:
# 1) we have 1000 cells and 2000 genes (2000 dimensions). we reduced the 2000 dimensions to 50 using PCA (50 Principal Components or 50 PCs)
# 2) now in clustering, we define the number of clusters (n_clusters). For our example, n_clusters = 3. we place these 3 centroid points
#    (representing 3 clusters) in the 50D space randomly
# 3) we place all 1000 data points (representing the 1000 cells) in this 50D space at the same time.
# 4) the k-means algo calculates the distance of each cell (data point) to these 3 centroids. To whichever centroid this cell point's distance 
#    is minimum to, the cell point becomes a part of that cluster the centroid represents. This happens simultaneously for all 1000 cell data points
# 5) now for each cluster, a new centroid is calculated based on the mean of all the cell data points in that cluster. This mean is calculated for
#    each of the 50 PCs (for each cell data point and each PC, let's say PC1, mean = sum of all PC1 values for that cell data point/1000), 
#    and then that new centroid point is obtained.
# 6) as a result of this new centroid calculation, sometimes cluster switching may happen for some cell data points.

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = kmeans.fit_predict(pca_results)

clusters

clusters.shape

# 5. Visualization (Using the first two PCs)
plt.figure(figsize=(10, 6))
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=clusters, cmap='viridis', s=10, alpha=0.7)
plt.colorbar(label='Assigned Cell Cluster')
plt.title("Clustering of 1,000 Cells based on 2,000 Gene Expressions")
plt.xlabel("PC1 (Variation in Marker Genes)")
plt.ylabel("PC2 (Variation in Marker Genes)")
plt.show()

### Observations and inference from the graph:
#### 1. 3 distinct groupings of clusters are visible, clearly separated from each other
#### 2. Each cluster represents a unique cell phenotype, eg.: Cluster 1 may be Neurons, since they share a unique gene expression signature
#### 3. Clear separation of clusters signifies the 150 marker genes "turned on" were the primary drivers of PC1 and PC2. The rest 1850 contributed almost nothing to the shape of the graph
