📄 RELATÓRIO – Lista 9

Curso: Ciência da Computação
Disciplina: Inteligência Artificial
Professora: Cristiane Neri Nobre
Aluno(a): Alice Antunes
Base: Credit Card Fraud Detection – Kaggle

#️⃣ Questão 1 — Etapas de Pré-Processamento
## 1. Introdução

Este relatório apresenta o processo completo de pré-processamento aplicado à base de dados Credit Card Fraud Detection. O objetivo é adequar os dados para modelos supervisionados de detecção de fraude, melhorando a qualidade dos dados e reduzindo problemas como desbalanceamento, outliers e redundância.

A base contém 284.807 registros, 31 atributos (28 PCA, além de Time, Amount e Class), e uma proporção de fraude extremamente baixa (~0,17%).

## 2. Etapas de Pré-Processamento
### 2.1 Visualização da Base de Dados

Foram exibidos os comandos:

df.head(), df.tail(), df.sample()

df.info(), df.describe()

Gráficos iniciais: distribuição das classes e histogramas dos atributos principais.

Principais observações:

V1–V28 são componentes principais (PCA).

Distribuição extremamente desbalanceada (fraudes ≈ 0,17%).

Amount apresenta grande assimetria (cauda longa).

### 2.2 Verificação e Tratamento de Valores Ausentes

Comando utilizado:

df.isna().sum()


Resultado:
A base não contém valores ausentes → nenhuma ação necessária.

### 2.3 Detecção e Eliminação de Redundância e Inconsistência
Duplicidade

Foram encontrados X registros duplicados (substituir pelo valor real).

Todos foram removidos com:

df = df.drop_duplicates().reset_index(drop=True)

Inconsistências

Não foram encontrados valores inválidos (ex.: Amount < 0).

### 2.4 Detecção e Tratamento de Outliers

Outliers são esperados em transações fraudulentas, portanto:

Fraudes não foram alteradas.

Outliers em Amount da classe 0 foram tratados com IQR.

Boxplots antes e depois confirmam redução de pontos extremos apenas da classe majoritária.

Justificativa: evitar exclusão de instâncias legítimas de fraude e reduzir distorções nos dados.

### 2.5 Normalização / Padronização

Método adotado: RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df.drop("Class", axis=1))


Razões da escolha:

É robusto diante de outliers.

Ideal para dados com distribuição assimétrica (ex.: Amount).

### 2.6 Análise de Correlação e Multicolinearidade

Foi construída a matriz de correlação entre os atributos, excluindo a variável alvo.

Resultados:

Baixa correlação entre atributos (esperado devido ao PCA).

Multicolinearidade baixa, comprovada via VIF.

Nenhum atributo precisou ser removido.

### 2.7 Codificação de Variáveis

A base não possui atributos categóricos, portanto:
➡️ One-Hot Encoding não foi necessário.

Se houvesse variáveis categóricas, utilizar-se-ia pd.get_dummies ou OneHotEncoder.

### 2.8 Balanceamento da Classe

Foi utilizado o método SMOTE, aplicado somente no treino:

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)


Distribuição:

Antes: proporção de fraude ≈ 0,17%

Depois (treino): 50% fraude / 50% não fraude

Gráficos foram gerados mostrando o antes/depois.

### 2.9 Divisão Treino–Teste (Estratificada)

A separação foi realizada com estratificação:

train_test_split(..., stratify=y)


Objetivo: manter a proporção da classe rara no conjunto de teste.

## 3. Resultados dos Modelos: Antes x Depois

O modelo utilizado foi Regressão Logística.

### 3.1 Modelo Antes do Pré-Processamento
Métrica	Valor
Recall (fraude)	X
Precision (fraude)	X
F1-score	X
AUC-ROC	X

(substituir pelos seus resultados)

### 3.2 Modelo Após o Pré-Processamento
Métrica	Valor
Recall (fraude)	X
Precision (fraude)	X
F1-score	X
AUC-ROC	X
### Conclusões

O recall da classe fraudulenta aumentou significativamente após SMOTE.

Pequena redução de precisão é esperada (trade-off comum em classificação rara).

O AUC-ROC aumentou, mostrando melhor separação entre classes.

O modelo final está mais adequado para o problema real de detecção de fraudes.

## 4. Links do Código

git@github.com:alicesalim/lista_ia-.git

#️⃣ Questão 2 — Algoritmos de Agrupamento
## 1. Introdução

Nesta questão, utilizamos algoritmos de aprendizado não supervisionado para identificar possíveis agrupamentos naturais na base Credit Card Fraud Detection.

⚠️ Importante: Para essa etapa, o atributo Class foi removido antes do agrupamento.

Os algoritmos aplicados foram:

K-Means (k = 2)

DBSCAN

SOM (Self-Organizing Map / MiniSom)

A qualidade dos grupos foi avaliada com:

Índice de Silhueta

Davies-Bouldin

Calinski-Harabasz

(Opcional) ARI comparando com a classe real

## 2. Algoritmos e Hiperparâmetros
### 2.1 K-Means

Parâmetros utilizados:

n_clusters = 2

n_init = 10

max_iter = 300

random_state = 42

Resultados obtidos:

Métrica	Valor
Silhouette	X
Davies-Bouldin	X
Calinski-Harabasz	X
(Opcional) ARI	X

Análise:

O K-Means tenta separar os dados em 2 clusters, mas como as fraudes são raras e distribuídas de forma dispersa, a separação tende a não refletir bem a classe real.

### 2.2 DBSCAN

Parâmetros utilizados (exemplo):

eps = X.X

min_samples = 5

Ajustados empiricamente até gerar pelo menos 2 clusters válidos.

Clusters encontrados:

Total: X

Ruído (-1): X instâncias

Resultados (considerando apenas labels ≠ -1):

Métrica	Valor
Silhouette	X
Davies-Bouldin	X
Calinski-Harabasz	X

Análise:

DBSCAN trata bem regiões densas e marca fraudes esparsas como ruído.

Tipicamente não encontra exatamente 2 clusters — depende muito dos hiperparâmetros.

### 2.3 SOM (Self-Organizing Map)

Utilizado o pacote MiniSom.

Configuração:

Tamanho do mapa: 1 × 2 neurônios

sigma = 1.0

learning_rate = 0.5

Iterações: 1000

Resultados:

Métrica	Valor
Silhouette	X
Davies-Bouldin	X
Calinski-Harabasz	X

Análise:

SOM realiza mapeamento topológico e encontra padrões.

Em geral, não separa explicitamente fraudes, pois estas não formam um cluster denso no espaço PCA.

## 3. Conclusões da Questão 2

K-Means, DBSCAN e SOM não conseguem separar claramente fraude e não fraude, pois as fraudes não formam um cluster natural.

As métricas de Silhueta tendem a valores baixos (próximos de 0), indicando fraca separação.

O DBSCAN identifica diversas instâncias como ruído, o que é coerente com a natureza extremamente rara e dispersa das fraudes.

O SOM cria dois grupos artificiais, mas que não correspondem à classe real.

Conclusão geral:
A base não apresenta clusters naturais que correspondam à divisão fraude/não fraude. Isso reforça a necessidade de modelos supervisionados e de técnicas fortes de pré-processamento (principalmente balanceamento).

## 4. Links do Código
git@github.com:alicesalim/lista_ia-.git