import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns", None)

# 1. CARREGAR A BASE
df_raw = pd.read_csv("creditcard.csv")

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

# 5. NORMALIZAÇÃO / PADRONIZAÇÃO (RobustScaler)
X_raw = df_raw.drop("Class", axis=1)
y_raw = df_raw["Class"]

X_pp = df_pp.drop("Class", axis=1)
y_pp = df_pp["Class"]

# 6. ANÁLISE DA CLASSE (DESBALANCEAMENTO)
print("\nDistribuição da classe (base original):")
print(y_raw.value_counts())
print((y_raw.value_counts(normalize=True) * 100).round(4))

# 7. DIVISÃO TREINO/TESTE (ESTRATIFICADA) - BASE ORIGINAL (BASELINE)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_raw,
    y_raw,
    test_size=0.2,
    stratify=y_raw,
    random_state=42
)

print("\nDimensões baseline - treino/teste:", X_train_b.shape, X_test_b.shape)

# 8. ESCALONAMENTO BASELINE (RobustScaler)
scaler_b = RobustScaler()
X_train_b_scaled = scaler_b.fit_transform(X_train_b)
X_test_b_scaled = scaler_b.transform(X_test_b)

# 9. MODELO BASELINE (SEM PRÉ-PROCESSAMENTO COMPLETO)
log_reg_b = LogisticRegression(max_iter=1000, n_jobs=-1)
log_reg_b.fit(X_train_b_scaled, y_train_b)

y_pred_b = log_reg_b.predict(X_test_b_scaled)
y_prob_b = log_reg_b.predict_proba(X_test_b_scaled)[:, 1]

print("\n=== RESULTADOS BASELINE (ANTES DO PRÉ-PROCESSAMENTO COMPLETO) ===")
print(classification_report(y_test_b, y_pred_b, digits=4))
print("AUC-ROC baseline:", roc_auc_score(y_test_b, y_prob_b))

# 10. DIVISÃO TREINO/TESTE (ESTRATIFICADA) - PÓS-PRÉ-PROCESSAMENTO
X_train, X_test, y_train, y_test = train_test_split(
    X_pp,
    y_pp,
    test_size=0.2,
    stratify=y_pp,
    random_state=42
)

print("\nDimensões pós-PP - treino/teste:", X_train.shape, X_test.shape)

# 11. ESCALONAMENTO PÓS-PRÉ-PROCESSAMENTO (RobustScaler)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 12. BALANCEAMENTO COM SMOTE
print("\nDistribuição original no treino pós-PP:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("\nDistribuição balanceada no treino (SMOTE):")
print(y_train_bal.value_counts())

# 13. MODELO PÓS-PRÉ-PROCESSAMENTO COMPLETO
log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
log_reg.fit(X_train_bal, y_train_bal)

y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n=== RESULTADOS APÓS PRÉ-PROCESSAMENTO COMPLETO ===")
print(classification_report(y_test, y_pred, digits=4))
print("AUC-ROC pós-PP:", roc_auc_score(y_test, y_prob))

# 14. VISUALIZAÇÕES
plt.figure(figsize=(5,4))
sns.countplot(x=y_raw)
plt.title("Distribuição da Classe - Base Original")
plt.show()

plt.figure(figsize=(5,4))
sns.boxplot(x=df_pp["Amount"])
plt.title("Boxplot Amount - Após Tratamento de Outliers (Classe 0)")
plt.show()
