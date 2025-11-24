import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from minisom import MiniSom

plt.style.use('default')

# 1. CARREGAR BASE E PREPARAR X (SEM CLASS)
try:
    df = pd.read_csv("creditcard.csv")
    print("Dataset real carregado com sucesso!")
except FileNotFoundError:
    try:
        df = pd.read_csv("creditcard_simulated.csv")
        print("Usando dataset simulado criado na Questão 1.")
    except FileNotFoundError:
        print("Erro: Nenhum dataset encontrado. Execute Q1.py primeiro.")
        exit()

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

# 5. COMPARAÇÃO DE ALGORITMOS E VISUALIZAÇÕES
print("\n" + "="*60)
print("           RESUMO COMPARATIVO DOS ALGORITMOS")
print("="*60)

# Criando tabela comparativa
results_df = pd.DataFrame({
    'Algoritmo': ['K-Means', 'DBSCAN', 'SOM'],
    'Silhouette': [sil_km, sil_db if 'sil_db' in locals() else np.nan, sil_som],
    'Davies-Bouldin': [db_km, db_db if 'db_db' in locals() else np.nan, db_som],
    'Calinski-Harabasz': [ch_km, ch_db if 'ch_db' in locals() else np.nan, ch_som],
    'ARI (vs Class)': [ari_km, ari_db if 'ari_db' in locals() else np.nan, ari_som]
})

print(results_df.to_string(index=False, float_format='%.4f'))

print("\nInterpretação das métricas:")
print("• Silhouette: [-1, 1] - Maior é melhor (> 0.5 = boa separação)")
print("• Davies-Bouldin: [0, ∞) - Menor é melhor (< 1.0 = boa separação)")  
print("• Calinski-Harabasz: [0, ∞) - Maior é melhor")
print("• ARI: [-1, 1] - Maior é melhor (1.0 = perfeita concordância)")

# 6. VISUALIZAÇÕES DOS CLUSTERS

# Redução de dimensionalidade para visualização
print("\n=== GERANDO VISUALIZAÇÕES ===")
print("Reduzindo dimensionalidade para visualização...")

# PCA para 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_clust)
print(f"Variância explicada pelo PCA: {pca.explained_variance_ratio_.sum():.3f}")

# t-SNE para 2D (em uma amostra menor para performance)
sample_size_tsne = min(5000, len(X_clust))
idx_tsne = np.random.choice(len(X_clust), size=sample_size_tsne, replace=False)
X_tsne_sample = X_clust[idx_tsne]
y_tsne_sample = y_clust.iloc[idx_tsne]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_tsne_sample)

# Visualização dos clusters
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Linha 1: PCA
# Classe real
scatter1 = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_clust, 
                            cmap='RdYlBu', alpha=0.6, s=20)
axes[0,0].set_title('Classe Real (PCA)')
axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.colorbar(scatter1, ax=axes[0,0])

# K-Means
scatter2 = axes[0,1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km, 
                            cmap='viridis', alpha=0.6, s=20)
axes[0,1].set_title('K-Means (PCA)')
axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.colorbar(scatter2, ax=axes[0,1])

# DBSCAN
scatter3 = axes[0,2].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_db, 
                            cmap='plasma', alpha=0.6, s=20)
axes[0,2].set_title('DBSCAN (PCA)')
axes[0,2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[0,2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.colorbar(scatter3, ax=axes[0,2])

# SOM
scatter4 = axes[0,3].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_som, 
                            cmap='coolwarm', alpha=0.6, s=20)
axes[0,3].set_title('SOM (PCA)')
axes[0,3].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[0,3].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.colorbar(scatter4, ax=axes[0,3])

# Linha 2: t-SNE (amostra)
# Classe real
scatter5 = axes[1,0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne_sample, 
                            cmap='RdYlBu', alpha=0.6, s=20)
axes[1,0].set_title('Classe Real (t-SNE)')
axes[1,0].set_xlabel('t-SNE 1')
axes[1,0].set_ylabel('t-SNE 2')
plt.colorbar(scatter5, ax=axes[1,0])

# K-Means
labels_km_sample = labels_km[idx_tsne]
scatter6 = axes[1,1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_km_sample, 
                            cmap='viridis', alpha=0.6, s=20)
axes[1,1].set_title('K-Means (t-SNE)')
axes[1,1].set_xlabel('t-SNE 1')
axes[1,1].set_ylabel('t-SNE 2')
plt.colorbar(scatter6, ax=axes[1,1])

# DBSCAN
labels_db_sample = labels_db[idx_tsne]
scatter7 = axes[1,2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_db_sample, 
                            cmap='plasma', alpha=0.6, s=20)
axes[1,2].set_title('DBSCAN (t-SNE)')
axes[1,2].set_xlabel('t-SNE 1')
axes[1,2].set_ylabel('t-SNE 2')
plt.colorbar(scatter7, ax=axes[1,2])

# SOM
labels_som_sample = labels_som[idx_tsne]
scatter8 = axes[1,3].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_som_sample, 
                            cmap='coolwarm', alpha=0.6, s=20)
axes[1,3].set_title('SOM (t-SNE)')
axes[1,3].set_xlabel('t-SNE 1')
axes[1,3].set_ylabel('t-SNE 2')
plt.colorbar(scatter8, ax=axes[1,3])

plt.tight_layout()
plt.savefig('clustering_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. GRÁFICO DE BARRAS COM AS MÉTRICAS
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Preparando dados (tratando NaN)
algorithms = ['K-Means', 'DBSCAN', 'SOM']
silhouette_scores = [sil_km, sil_db if 'sil_db' in locals() else 0, sil_som]
davies_bouldin_scores = [db_km, db_db if 'db_db' in locals() else 0, db_som]
calinski_scores = [ch_km, ch_db if 'ch_db' in locals() else 0, ch_som]
ari_scores = [ari_km, ari_db if 'ari_db' in locals() else 0, ari_som]

# Gráfico 1: Silhouette
bars1 = axes[0,0].bar(algorithms, silhouette_scores, color=['blue', 'orange', 'green'], alpha=0.7)
axes[0,0].set_title('Índice de Silhueta (Maior = Melhor)')
axes[0,0].set_ylabel('Silhouette Score')
axes[0,0].grid(True, alpha=0.3)
for i, v in enumerate(silhouette_scores):
    axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Gráfico 2: Davies-Bouldin
bars2 = axes[0,1].bar(algorithms, davies_bouldin_scores, color=['blue', 'orange', 'green'], alpha=0.7)
axes[0,1].set_title('Davies-Bouldin (Menor = Melhor)')
axes[0,1].set_ylabel('Davies-Bouldin Score')
axes[0,1].grid(True, alpha=0.3)
for i, v in enumerate(davies_bouldin_scores):
    axes[0,1].text(i, v + max(davies_bouldin_scores)*0.02, f'{v:.3f}', ha='center', va='bottom')

# Gráfico 3: Calinski-Harabasz
bars3 = axes[1,0].bar(algorithms, calinski_scores, color=['blue', 'orange', 'green'], alpha=0.7)
axes[1,0].set_title('Calinski-Harabasz (Maior = Melhor)')
axes[1,0].set_ylabel('Calinski-Harabasz Score')
axes[1,0].grid(True, alpha=0.3)
for i, v in enumerate(calinski_scores):
    axes[1,0].text(i, v + max(calinski_scores)*0.02, f'{v:.0f}', ha='center', va='bottom')

# Gráfico 4: ARI
bars4 = axes[1,1].bar(algorithms, ari_scores, color=['blue', 'orange', 'green'], alpha=0.7)
axes[1,1].set_title('Adjusted Rand Index (Maior = Melhor)')
axes[1,1].set_ylabel('ARI Score')
axes[1,1].grid(True, alpha=0.3)
for i, v in enumerate(ari_scores):
    axes[1,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('clustering_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. ANÁLISE DOS HIPERPARÂMETROS DO DBSCAN
print("\n=== OTIMIZAÇÃO DE HIPERPARÂMETROS DBSCAN ===")
print("Testando diferentes valores de eps e min_samples...")

eps_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
min_samples_range = [3, 5, 10, 15]

best_silhouette = -1
best_params = {}
dbscan_results = []

for eps in eps_range:
    for min_samples in min_samples_range:
        dbscan_temp = DBSCAN(eps=eps, min_samples=min_samples)
        labels_temp = dbscan_temp.fit_predict(X_clust)
        
        # Verificar se encontrou clusters válidos
        valid_labels = labels_temp[labels_temp != -1]
        if len(np.unique(valid_labels)) > 1:
            sil_temp = silhouette_score(X_clust[labels_temp != -1], valid_labels)
            n_clusters = len(np.unique(valid_labels))
            n_noise = np.sum(labels_temp == -1)
            
            dbscan_results.append({
                'eps': eps,
                'min_samples': min_samples,
                'silhouette': sil_temp,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': n_noise / len(labels_temp)
            })
            
            if sil_temp > best_silhouette:
                best_silhouette = sil_temp
                best_params = {'eps': eps, 'min_samples': min_samples}

if dbscan_results:
    dbscan_df = pd.DataFrame(dbscan_results)
    print("\nMelhores 5 configurações DBSCAN:")
    print(dbscan_df.nlargest(5, 'silhouette').to_string(index=False, float_format='%.4f'))
    
    print(f"\nMelhor configuração: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
    print(f"Silhouette: {best_silhouette:.4f}")

# 9. CONCLUSÕES FINAIS
print("\n" + "="*70)
print("                        CONCLUSÕES FINAIS")
print("="*70)

print("\n1. QUALIDADE DOS AGRUPAMENTOS:")
print(f"   • K-Means: Silhouette = {sil_km:.4f}")
if 'sil_db' in locals():
    print(f"   • DBSCAN:  Silhouette = {sil_db:.4f}")
else:
    print("   • DBSCAN:  Não encontrou clusters válidos com os parâmetros padrão")
print(f"   • SOM:     Silhouette = {sil_som:.4f}")

print("\n2. CORRESPONDÊNCIA COM A CLASSE REAL (ARI):")
print(f"   • K-Means: ARI = {ari_km:.4f}")
if 'ari_db' in locals():
    print(f"   • DBSCAN:  ARI = {ari_db:.4f}")
else:
    print("   • DBSCAN:  Não calculado (clusters insuficientes)")
print(f"   • SOM:     ARI = {ari_som:.4f}")

print("\n3. INTERPRETAÇÃO:")
print("   • Valores de Silhouette próximos de 0 indicam clusters pouco separados")
print("   • ARI baixo sugere que os clusters não correspondem às classes reais")
print("   • Isso é esperado: fraudes são raras e dispersas, não formam clusters naturais")
print("   • Confirma a necessidade de métodos supervisionados para detecção de fraude")

print("\n✓ Todas as visualizações foram salvas como arquivos PNG")
