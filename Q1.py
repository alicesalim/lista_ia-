import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option("display.max_columns", None)
plt.style.use('default')  # Para gráficos mais bonitos

# 1. CARREGAR A BASE
# Para demonstração, vou criar um dataset simulado semelhante ao Credit Card Fraud
# (substitua por df_raw = pd.read_csv("creditcard.csv") quando tiver o arquivo real)

def create_sample_creditcard_data():
    # Criando dados simulados semelhantes ao dataset de fraude de cartão
    np.random.seed(42)
    n_samples = 10000
    n_fraud = 200  # ~2% de fraudes
    
    # Simulando V1-V28 (componentes PCA)
    X_normal, _ = make_classification(
        n_samples=n_samples-n_fraud,
        n_features=28,
        n_informative=20,
        n_redundant=8,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_fraud, _ = make_classification(
        n_samples=n_fraud,
        n_features=28,
        n_informative=20,
        n_redundant=8,
        n_clusters_per_class=1,
        random_state=123
    )
    
    # Mudando a distribuição das fraudes para serem mais extremas
    X_fraud = X_fraud * 2 + np.random.normal(0, 0.5, X_fraud.shape)
    
    # Combinando dados
    X_combined = np.vstack([X_normal, X_fraud])
    y_combined = np.hstack([np.zeros(n_samples-n_fraud), np.ones(n_fraud)])
    
    # Criando DataFrame
    columns = [f'V{i}' for i in range(1, 29)]
    df = pd.DataFrame(X_combined, columns=columns)
    
    # Adicionando Time e Amount
    df['Time'] = np.random.uniform(0, 172800, n_samples)  # 2 dias em segundos
    df['Amount'] = np.random.lognormal(mean=3, sigma=1.5, size=n_samples)
    
    # Fraudes tendem a ter valores menores
    fraud_mask = y_combined == 1
    df.loc[fraud_mask, 'Amount'] = np.random.lognormal(mean=2, sigma=1, size=n_fraud)
    
    df['Class'] = y_combined
    
    # Adicionando alguns duplicados e outliers para demonstração
    df = pd.concat([df, df.sample(50, random_state=42)], ignore_index=True)
    
    return df

try:
    df_raw = pd.read_csv("creditcard.csv")
    print("Dataset real carregado com sucesso!")
except FileNotFoundError:
    print("Arquivo creditcard.csv não encontrado. Usando dataset simulado para demonstração.")
    df_raw = create_sample_creditcard_data()
    # Salvando o dataset simulado para uso consistente
    df_raw.to_csv("creditcard_simulated.csv", index=False)

print("Dimensão inicial da base:", df_raw.shape)
print(df_raw.head())
print(df_raw.info())

print(df_raw["Class"].value_counts(normalize=True) * 100)

# 2. VALORES AUSENTES
print("\nValores ausentes por coluna:")
print(df_raw.isna().sum())

# 3. DUPLICADOS E INCONSISTÊNCIAS
num_dup = df_raw.duplicated().sum()
print(f"\nNúmero de linhas duplicadas: {num_dup}")

df = df_raw.drop_duplicates().reset_index(drop=True)
print("Dimensão após remoção de duplicados:", df.shape)

print("\nValores Amount < 0:", (df["Amount"] < 0).sum())
print("Valores Time < 0:", (df["Time"] < 0).sum())

# 4. OUTLIERS (tratando apenas Amount na classe 0 com IQR)
df_out = df.copy()

q1 = df_out.loc[df_out["Class"] == 0, "Amount"].quantile(0.25)
q3 = df_out.loc[df_out["Class"] == 0, "Amount"].quantile(0.75)
iqr = q3 - q1

lim_inf = q1 - 1.5 * iqr
lim_sup = q3 + 1.5 * iqr

print("\nLimites IQR Amount (classe 0):", lim_inf, lim_sup)

mask_ok = (df_out["Class"] == 1) | (
    (df_out["Amount"] >= lim_inf) & (df_out["Amount"] <= lim_sup)
)

df_pp = df_out[mask_ok].reset_index(drop=True)
print("Dimensão após tratar outliers de Amount (classe 0):", df_pp.shape)

# 5. ANÁLISE DE CORRELAÇÃO E MULTICOLINEARIDADE
print("\n=== ANÁLISE DE CORRELAÇÃO E MULTICOLINEARIDADE ===")

X_corr = df_pp.drop("Class", axis=1)

# Matriz de correlação
correlation_matrix = X_corr.corr()
print("Correlações mais altas (> 0.7):")
high_corr = np.where(np.abs(correlation_matrix) > 0.7)
high_corr_pairs = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y]) 
                   for x, y in zip(*high_corr) if x != y and x < y]

if high_corr_pairs:
    for var1, var2, corr in high_corr_pairs[:10]:  # Mostra apenas as 10 primeiras
        print(f"{var1} - {var2}: {corr:.4f}")
else:
    print("Nenhuma correlação > 0.7 encontrada entre variáveis diferentes.")

# Visualização da matriz de correlação (apenas para algumas variáveis por questão de espaço)
plt.figure(figsize=(12, 10))
# Selecionando algumas variáveis para visualização
vars_to_plot = X_corr.columns[:15]  # Primeiras 15 variáveis + Time e Amount
subset_corr = correlation_matrix.loc[vars_to_plot, vars_to_plot]

sns.heatmap(subset_corr, annot=False, cmap='coolwarm', center=0, 
            square=True, cbar_kws={'label': 'Correlação'})
plt.title("Matriz de Correlação (Amostra de Variáveis)")
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise de Multicolinearidade usando VIF
print("\nAnálise de Multicolinearidade (VIF):")
print("(Valores VIF > 10 indicam multicolinearidade alta)")

# Calculando VIF para algumas variáveis (muito custoso para todas)
sample_vars = X_corr.columns[-5:]  # Últimas 5 variáveis
X_vif = X_corr[sample_vars]

vif_data = pd.DataFrame()
vif_data["Variável"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) 
                   for i in range(len(X_vif.columns))]

print(vif_data)
print("Interpretação: VIF < 5 (baixa), 5-10 (moderada), > 10 (alta multicolinearidade)")

# 6. CODIFICAÇÃO DE VARIÁVEIS
print("\n=== CODIFICAÇÃO DE VARIÁVEIS ===")
print("Verificando se há variáveis categóricas que precisam de codificação...")

categorical_vars = X_corr.select_dtypes(include=['object', 'category']).columns
print(f"Variáveis categóricas encontradas: {list(categorical_vars)}")

if len(categorical_vars) == 0:
    print("✓ Nenhuma variável categórica encontrada. One-Hot Encoding não necessário.")
    print("✓ Todas as variáveis já são numéricas.")
else:
    print("Aplicando One-Hot Encoding nas variáveis categóricas:")
    # Aqui aplicaria pd.get_dummies() se necessário
    for var in categorical_vars:
        print(f"  - {var}: {X_corr[var].unique()}")

# 7. NORMALIZAÇÃO / PADRONIZAÇÃO (RobustScaler)
X_raw = df_raw.drop("Class", axis=1)
y_raw = df_raw["Class"]

X_pp = df_pp.drop("Class", axis=1)
y_pp = df_pp["Class"]

# 8. ANÁLISE DA CLASSE (DESBALANCEAMENTO)
print("\nDistribuição da classe (base original):")
print(y_raw.value_counts())
print((y_raw.value_counts(normalize=True) * 100).round(4))

# 9. DIVISÃO TREINO/TESTE (ESTRATIFICADA) - BASE ORIGINAL (BASELINE)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_raw,
    y_raw,
    test_size=0.2,
    stratify=y_raw,
    random_state=42
)

print("\nDimensões baseline - treino/teste:", X_train_b.shape, X_test_b.shape)

# 10. ESCALONAMENTO BASELINE (RobustScaler)
scaler_b = RobustScaler()
X_train_b_scaled = scaler_b.fit_transform(X_train_b)
X_test_b_scaled = scaler_b.transform(X_test_b)

# 11. MODELO BASELINE (SEM PRÉ-PROCESSAMENTO COMPLETO)
log_reg_b = LogisticRegression(max_iter=1000, n_jobs=-1)
log_reg_b.fit(X_train_b_scaled, y_train_b)

y_pred_b = log_reg_b.predict(X_test_b_scaled)
y_prob_b = log_reg_b.predict_proba(X_test_b_scaled)[:, 1]

print("\n=== RESULTADOS BASELINE (ANTES DO PRÉ-PROCESSAMENTO COMPLETO) ===")
print(classification_report(y_test_b, y_pred_b, digits=4))
print("AUC-ROC baseline:", roc_auc_score(y_test_b, y_prob_b))

# 12. DIVISÃO TREINO/TESTE (ESTRATIFICADA) - PÓS-PRÉ-PROCESSAMENTO
X_train, X_test, y_train, y_test = train_test_split(
    X_pp,
    y_pp,
    test_size=0.2,
    stratify=y_pp,
    random_state=42
)

print("\nDimensões pós-PP - treino/teste:", X_train.shape, X_test.shape)

# 13. ESCALONAMENTO PÓS-PRÉ-PROCESSAMENTO (RobustScaler)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 14. BALANCEAMENTO COM SMOTE
print("\nDistribuição original no treino pós-PP:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("\nDistribuição balanceada no treino (SMOTE):")
print(y_train_bal.value_counts())

# 15. MODELO PÓS-PRÉ-PROCESSAMENTO COMPLETO
log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
log_reg.fit(X_train_bal, y_train_bal)

y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n=== RESULTADOS APÓS PRÉ-PROCESSAMENTO COMPLETO ===")
print(classification_report(y_test, y_pred, digits=4))
print("AUC-ROC pós-PP:", roc_auc_score(y_test, y_prob))

# 16. VISUALIZAÇÕES ADICIONAIS
print("\n=== GERANDO VISUALIZAÇÕES ===")

# 1. Distribuição das classes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Distribuição original
axes[0].pie(y_raw.value_counts(), labels=['Normal (0)', 'Fraude (1)'], 
           autopct='%1.2f%%', startangle=90, colors=['lightblue', 'salmon'])
axes[0].set_title("Distribuição Original das Classes")

# Distribuição balanceada (treino)
if 'y_train_bal' in locals():
    axes[1].pie(pd.Series(y_train_bal).value_counts(), labels=['Normal (0)', 'Fraude (1)'], 
               autopct='%1.2f%%', startangle=90, colors=['lightblue', 'salmon'])
    axes[1].set_title("Distribuição Balanceada (Treino - SMOTE)")

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Boxplot Amount: antes e depois do tratamento de outliers
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Antes do tratamento
axes[0].boxplot([df_raw[df_raw["Class"] == 0]["Amount"], 
                df_raw[df_raw["Class"] == 1]["Amount"]], 
                labels=['Normal', 'Fraude'])
axes[0].set_title("Amount - Antes do Tratamento (Outliers)")
axes[0].set_ylabel("Amount")
axes[0].set_yscale('log')  # Escala log para melhor visualização

# Depois do tratamento
axes[1].boxplot([df_pp[df_pp["Class"] == 0]["Amount"], 
                df_pp[df_pp["Class"] == 1]["Amount"]], 
                labels=['Normal', 'Fraude'])
axes[1].set_title("Amount - Após Tratamento de Outliers")
axes[1].set_ylabel("Amount")
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('amount_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Histograma da variável Amount
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df_raw[df_raw["Class"] == 0]["Amount"], bins=50, alpha=0.7, 
         label='Normal', color='blue', density=True)
plt.hist(df_raw[df_raw["Class"] == 1]["Amount"], bins=50, alpha=0.7, 
         label='Fraude', color='red', density=True)
plt.xlabel("Amount")
plt.ylabel("Densidade")
plt.title("Distribuição Amount - Original")
plt.legend()
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.hist(df_pp[df_pp["Class"] == 0]["Amount"], bins=50, alpha=0.7, 
         label='Normal', color='blue', density=True)
plt.hist(df_pp[df_pp["Class"] == 1]["Amount"], bins=50, alpha=0.7, 
         label='Fraude', color='red', density=True)
plt.xlabel("Amount")
plt.ylabel("Densidade")
plt.title("Distribuição Amount - Pós-tratamento")
plt.legend()
plt.yscale('log')

plt.tight_layout()
plt.savefig('amount_histograms.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Gráfico de comparação de métricas
if 'y_pred_b' in locals() and 'y_pred' in locals():
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Calculando métricas para comparação
    metrics_baseline = {
        'Precision': precision_score(y_test_b, y_pred_b),
        'Recall': recall_score(y_test_b, y_pred_b),
        'F1-Score': f1_score(y_test_b, y_pred_b),
        'AUC-ROC': roc_auc_score(y_test_b, y_prob_b)
    }
    
    metrics_processed = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_prob)
    }
    
    # Gráfico de barras comparativo
    x = range(len(metrics_baseline))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], list(metrics_baseline.values()), 
            width, label='Antes do Pré-processamento', alpha=0.8, color='lightcoral')
    plt.bar([i + width/2 for i in x], list(metrics_processed.values()), 
            width, label='Após Pré-processamento', alpha=0.8, color='lightgreen')
    
    plt.xlabel('Métricas')
    plt.ylabel('Valor')
    plt.title('Comparação de Performance: Antes vs Depois do Pré-processamento')
    plt.xticks(x, list(metrics_baseline.keys()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionando valores nas barras
    for i, (v1, v2) in enumerate(zip(metrics_baseline.values(), metrics_processed.values())):
        plt.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== RESUMO COMPARATIVO ===")
    print("BASELINE (sem pré-processamento completo):")
    for metric, value in metrics_baseline.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nAPÓS PRÉ-PROCESSAMENTO:")
    for metric, value in metrics_processed.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nMELHORIAS:")
    for metric in metrics_baseline.keys():
        improvement = metrics_processed[metric] - metrics_baseline[metric]
        print(f"  {metric}: {improvement:+.4f}")

print("\n✓ Todas as visualizações foram salvas como arquivos PNG")
