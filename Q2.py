import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

from minisom import MiniSom

# 1. CARREGAR BASE E PREPARAR X (SEM CLASS)
df = pd.read_csv("creditcard.csv")
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
sample_size = 50000 
if len(X_scaled) > sample_size:
    np.random.seed(42)
    idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)
    X_clust = X_scaled[idx]
    y_clust = y.iloc[idx].reset_index(drop=True)
else:
    X_clust = X_scaled
    y_clust = y.reset_index(drop=True)

print("Tamanho usado para clustering:", X_clust.shape)

# 2. KMEANS (k = 2)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_clust)

sil_km = silhouette_score(X_clust, labels_km)
db_km = davies_bouldin_score(X_clust, labels_km)
ch_km = calinski_harabasz_score(X_clust, labels_km)
ari_km = adjusted_rand_score(y_clust, labels_km)

print("\n=== KMEANS (k = 2) ===")
print("Silhouette:", sil_km)
print("Davies-Bouldin:", db_km)
print("Calinski-Harabasz:", ch_km)
print("Adjusted Rand Index (comparando com Class):", ari_km)

# 3. DBSCAN
dbscan = DBSCAN(eps=2.0, min_samples=5)  
labels_db = dbscan.fit_predict(X_clust)

unique_labels = np.unique(labels_db)
print("\nRótulos DBSCAN encontrados:", unique_labels)
print("Quantidade de clusters (sem contar -1):", len(unique_labels[unique_labels != -1]))

mask_valid = labels_db != -1
if len(np.unique(labels_db[mask_valid])) > 1:
    sil_db = silhouette_score(X_clust[mask_valid], labels_db[mask_valid])
    db_db = davies_bouldin_score(X_clust[mask_valid], labels_db[mask_valid])
    ch_db = calinski_harabasz_score(X_clust[mask_valid], labels_db[mask_valid])
    ari_db = adjusted_rand_score(y_clust[mask_valid], labels_db[mask_valid])

    print("\n=== DBSCAN (sem ruído) ===")
    print("Silhouette:", sil_db)
    print("Davies-Bouldin:", db_db)
    print("Calinski-Harabasz:", ch_db)
    print("Adjusted Rand Index:", ari_db)
else:
    print("DBSCAN não encontrou múltiplos clusters válidos para calcular métricas.")

# 4. SOM (Self-Organizing Map)
som = MiniSom(
    x=1,
    y=2,
    input_len=X_clust.shape[1],
    sigma=1.0,
    learning_rate=0.5,
    random_seed=42
)

som.random_weights_init(X_clust)
som.train_random(X_clust, 1000)  
winner_coords = np.array([som.winner(x) for x in X_clust])
labels_som = winner_coords[:, 1]

print("\nRótulos SOM (0/1):", np.unique(labels_som, return_counts=True))

sil_som = silhouette_score(X_clust, labels_som)
db_som = davies_bouldin_score(X_clust, labels_som)
ch_som = calinski_harabasz_score(X_clust, labels_som)
ari_som = adjusted_rand_score(y_clust, labels_som)

print("\n=== SOM (1x2) ===")
print("Silhouette:", sil_som)
print("Davies-Bouldin:", db_som)
print("Calinski-Harabasz:", ch_som)
print("Adjusted Rand Index:", ari_som)
