import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

df = pd.read_csv("creditcard.csv")
df.head()
df.info()
df.describe()
df["Class"].value_counts(normalize=True)

df.head()
df.tail()
df.sample(5, random_state=42)

df.info()
df.describe()

sns.countplot(x="Class", data=df)
plt.title("Distribuição da Classe (0 = não fraude, 1 = fraude)")
plt.show()

sns.histplot(df["Amount"], bins=50)
plt.title("Distribuição do valor das transações (Amount)")
plt.show()

df.isna().sum()
(df.isna().sum() / len(df)) * 100


df.duplicated().sum()

df = df.drop_duplicates().reset_index(drop=True)

(df["Amount"] < 0).sum(), (df["Time"] < 0).sum()

q1 = df.loc[df["Class"] == 0, "Amount"].quantile(0.25)
q3 = df.loc[df["Class"] == 0, "Amount"].quantile(0.75)
iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

mask_ok = (df["Class"] == 1) | ((df["Amount"] >= limite_inferior) & (df["Amount"] <= limite_superior))
df = df[mask_ok].reset_index(drop=True)

features = df.drop("Class", axis=1)
target = df["Class"]

scaler = RobustScaler()
features_scaled = scaler.fit_transform(features)

X = pd.DataFrame(features_scaled, columns=features.columns)
y = target.copy()

corr = df.drop("Class", axis=1).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Matriz de correlação entre atributos (sem Class)")
plt.show()


from statsmodels.stats.outliers_influence import variance_inflation_factor

X_num = df.drop(columns=["Class"])
vif_data = pd.DataFrame()
vif_data["feature"] = X_num.columns
vif_data["VIF"] = [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])]
vif_data

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_num = df.drop(columns=["Class"])
vif_data = pd.DataFrame()
vif_data["feature"] = X_num.columns
vif_data["VIF"] = [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])]
vif_data

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

y_train.value_counts(), y_train_bal.value_counts()

from sklearn.linear_model import LogisticRegression

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    df.drop("Class", axis=1),
    df["Class"],
    test_size=0.2,
    stratify=df["Class"],
    random_state=42
)

scaler_b = RobustScaler()
X_train_b_scaled = scaler_b.fit_transform(X_train_b)
X_test_b_scaled  = scaler_b.transform(X_test_b)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_b_scaled, y_train_b)

y_pred_b = log_reg.predict(X_test_b_scaled)
y_prob_b = log_reg.predict_proba(X_test_b_scaled)[:, 1]

print(classification_report(y_test_b, y_pred_b, digits=4))
print("AUC:", roc_auc_score(y_test_b, y_prob_b))

# Usa X_train_bal, y_train_bal, X_test, y_test da parte anterior
log_reg2 = LogisticRegression(max_iter=1000)
log_reg2.fit(X_train_bal, y_train_bal)

y_pred = log_reg2.predict(X_test)
y_prob = log_reg2.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, digits=4))
print("AUC:", roc_auc_score(y_test, y_prob))
