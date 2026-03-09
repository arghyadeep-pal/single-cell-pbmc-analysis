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

# Clustering without pre-processing and PCA

### CASE 1: Cell data points having similar range of values for genes (Lack of marker genes)

#### Observations:
##### 1. All cell data points end up being close to each other so after clustering, they just envelope each other.
##### 2. The visualization looks like a single envelope having 3 different slices (3 different colours for 3 clusters)
##### 3. Even if new centroids are calculated for all 3 clusters, they all end up very close to each other, signifying a global mean for the entire 1000 cells, making a single envelope again with all clusters merged into each other.
##### 4. Even if one random gene shows some degree of separation, it may be skipped because of the plot showing gene 1 vs gene 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Create dummy data: 1000 cells x 2000 genes
# Assume there are 3 real underlying groups
np.random.seed(42)
cells = 1000
genes = 2000
X = np.random.normal(0, 1, (cells, genes))

# --- CASE 1: Similar Ranges (Noise Genes) ---
# Clustering the raw data
labels_case1 = KMeans(n_clusters=3, n_init='auto').fit_predict(X)

# --- VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot Case 1: The Blob
ax1.scatter(X[:, 0], X[:, 1], c=labels_case1, cmap='viridis', s=15, alpha=0.6)
ax1.set_title("Case 1: Similar Ranges\n(Overlapping 'Blob' of Noise)")
ax1.set_xlabel("Gene 1 (Low Scale)")
ax1.set_ylabel("Gene 2 (Low Scale)")

plt.tight_layout()
plt.show()

### CASE 2: One gene (eg., Gene 1) has exceptionally higher value compared to other genes (This gene is the marker gene)

#### Observations:
##### 1. The gene1 value for each cell data point is exceptionally high (x10e3), so if we plot gene1 vs gene2, we can see all the cell data points scattered around the 1000, 2000, 3000, etc. values along the x-axis, with comparatively very smaller values along the y-axis.
##### 2. When new centroids are calculated, since they are based on Euclidean distances, they gravitate towards the exceptionally large gene1 values in the x-axis, forming bands of clusters along the x-axis.
##### 3. So ideally the clusters may form in this manner, cluster1 with gene1 (x-axis) values between 1000-2000, cluster2 with x-axis values between 2000-3000, cluster3 with x-axis values > 3000, and so on.
##### 4. If the marker gene was represented along y-axis, then the bands of clusters would be parallel to the y-axis.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Create dummy data: 1000 cells x 2000 genes
# Assume there are 3 real underlying groups
np.random.seed(42)
cells = 1000
genes = 2000
X = np.random.normal(0, 1, (cells, genes))

# --- CASE 2: One High-Range Marker Gene ---
X_high_scale = X.copy()
# Gene 0 is now a "marker" with values up to 5000
X_high_scale[:, 0] = np.random.uniform(0, 5000, cells) 
labels_case2 = KMeans(n_clusters=3, n_init='auto').fit_predict(X_high_scale)

# --- VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot Case 2: The Banded Stripes
ax2.scatter(X_high_scale[:, 0], X_high_scale[:, 1], c=labels_case2, cmap='viridis', s=15, alpha=0.6)
ax2.set_title("Case 2: One High-Range Gene\n(Vertical 'Banding' along X-axis)")
ax2.set_xlabel("Gene 1 (High Scale: 0-5000)")
ax2.set_ylabel("Gene 2 (Low Scale: 0-10)")

plt.tight_layout()
plt.show()

# Clustering without PCA

### CASE 1: Cell data points having similar range of values for genes (Lack of marker genes)

#### Observations:
##### 1. All cell data points end up being close to each other so after clustering, they just envelope each other.
##### 2. The visualization looks like a single envelope having 3 different slices (3 different colours for 3 clusters)
##### 3. Even if new centroids are calculated for all 3 clusters, they all end up very close to each other, signifying a global mean for the entire 1000 cells, making a single envelope again with all clusters merged into each other.
##### 4. Even if one random gene shows some degree of separation, it may be skipped because of the plot showing gene 1 vs gene 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Create dummy data: 1000 cells x 2000 genes
# Assume there are 3 real underlying groups
np.random.seed(42)
cells = 1000
genes = 2000
X = np.random.normal(0, 1, (cells, genes))

# --- PREPROCESSING
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

# --- CASE 1: Similar Ranges (Noise Genes) ---
# Clustering the raw data
labels_case1 = KMeans(n_clusters=3, n_init='auto').fit_predict(data_scaled)

# --- VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot Case 1: The Blob
ax1.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels_case1, cmap='viridis', s=15, alpha=0.6)
ax1.set_title("Case 1: Similar Ranges\n(Overlapping 'Blob' of Noise)")
ax1.set_xlabel("Gene 1 (Low Scale)")
ax1.set_ylabel("Gene 2 (Low Scale)")

plt.tight_layout()
plt.show()

### CASE 2: One gene (eg., Gene 1) has exceptionally higher value compared to other genes (This gene is the marker gene)

#### Observations:
##### 1. The plot looks very similar to 1st case, except the points are still a bit stretched out along the x-axis (due to the marker gene, gene1 expressed along x-axis.
##### 2. The visualization looks like a slightly stretched out envelope along x-axis, having 3 different colours for 3 clusters
##### 3. Even if new centroids are calculated for all 3 clusters, they all end up very close to each other, signifying a global mean for the entire 1000 cells, making a single envelope again with all clusters merged into each other.
##### 4. Even if one random gene shows some degree of separation, it may be skipped because of the plot showing gene 1 vs gene 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Create dummy data: 1000 cells x 2000 genes
# Assume there are 3 real underlying groups
np.random.seed(42)
cells = 1000
genes = 2000
X = np.random.normal(0, 1, (cells, genes))

# --- CASE 2: One High-Range Marker Gene ---
X_high_scale = X.copy()
# Gene 0 is now a "marker" with values up to 5000
X_high_scale[:, 0] = np.random.uniform(0, 5000, cells) 

# --- PREPROCESSING
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X_high_scale)

labels_case2 = KMeans(n_clusters=3, n_init='auto').fit_predict(data_scaled)

# --- VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot Case 2: The Banded Stripes
ax2.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels_case2, cmap='viridis', s=15, alpha=0.6)
ax2.set_title("Case 2: One High-Range Gene\n(Vertical 'Banding' along X-axis)")
ax2.set_xlabel("Gene 1 (High Scale: 0-5000)")
ax2.set_ylabel("Gene 2 (Low Scale: 0-10)")

plt.tight_layout()
plt.show()

### FINAL OBSERVATIONS FOR CLUSTERING WITH PREPROCESSING AND WITHOUT PCA

#### Observations:
##### 1. The k-means clustering works perfectly as per its logic
##### 2. The visualization is useless since it is showing an incomplete representation, without proper classified separation of data points.
##### 3. It is an incomplete picture since it is only showing the behaviour of only 2 genes from 2000 genes
