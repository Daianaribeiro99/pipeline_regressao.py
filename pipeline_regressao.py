import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Criando dados fictícios
dados = {
    "Combustivel": ["Gasolina", "Diesel", "Etanol", "Gasolina", "Diesel", "Etanol", "Gasolina"],
    "Idade": [5, 2, 8, 3, 6, 1, 4],
    "Quilometragem": [60000, 25000, 90000, 30000, 70000, 15000, 50000],
    "Preco": [30000, 50000, 20000, 45000, 28000, 55000, 35000]
}

# Criando um DataFrame
df = pd.DataFrame(dados)

# Separando as variáveis independentes (X) e dependente (y)
X = df.drop("Preco", axis=1)
y = df["Preco"]

# Separando os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo as transformações
transformador = ColumnTransformer([
    ("categorico", OneHotEncoder(), ["Combustivel"]),
    ("numerico", StandardScaler(), ["Idade", "Quilometragem"])
])

# Criando o pipeline
pipeline = Pipeline([
    ("preprocessamento", transformador),
    ("modelo", LinearRegression())
])

# Treinando o modelo
pipeline.fit(X_train, y_train)

# Fazendo previsões
y_pred = pipeline.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
