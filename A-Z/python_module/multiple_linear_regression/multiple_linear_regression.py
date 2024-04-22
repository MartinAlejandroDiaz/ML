# Regresión lineal Multiple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('../data/50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labellencoder_X = LabelEncoder()
x[:,3]=labellencoder_X.fit_transform(x[:,3])

print("//////////////////////////////")
print("Transformamos una variable unica no comparable a una dummy")
print("//////////////////////////////")

transformador_columnas = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [3])  # [0] es el índice de la columna que se va a codificar en one-hot
    ],
    remainder='passthrough'  # pasa las columnas no especificadas
)

# Ajusta y transforma tus datos usando el transformador de columnas
X_codificado = transformador_columnas.fit_transform(x)
print(X_codificado)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(X_train, X_test, y_train, y_test)

# Ajustar el modelo de regreción lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de testing
print("Predicción de los resultados en el conjunto de testing")
y_pred = regression.predict(X_test)
print(y_pred)

# Construir el moelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm
x = np.append(arr = x, values = np.ones((50,1)).astype(int), axis=1)

# endog es la variable a predecir
# exog es la matriz de caracteristicas
sl=.05
x_opt = x[:, [0, 1, 2, 3, 4]]
model = sm.OLS(y, x_opt)
regression_OLS = model.fit()
regression_OLS.summary()
print(regression_OLS.summary())