# Plantilla de preprocesado

# Como importar las librerias - Datos Categoricos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('../Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labellencoder_X = LabelEncoder()
x[:,0]=labellencoder_X.fit_transform(x[:,0])
print(x)

print("//////////////////////////////")
print("Transformamos una variable unica no comparable a una dummy")
print("//////////////////////////////")
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define el transformador de columnas
# Especifica la(s) columna(s) que deseas codificar en one-hot
transformador_columnas = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [0])  # [0] es el índice de la columna que se va a codificar en one-hot
    ],
    remainder='passthrough'  # pasa las columnas no especificadas
)

# Ajusta y transforma tus datos usando el transformador de columnas
X_codificado = transformador_columnas.fit_transform(x)
print(X_codificado)
print("//////////////////////////////")
print("Codificado de respuestas")
print("//////////////////////////////")
labellencoder_y = LabelEncoder()
y = labellencoder_y.fit_transform(y)
print(y)