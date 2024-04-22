# Plantilla de preprocesado

# Como importar las librerias - Datos Faltantes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('../Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Tratamiento de los NAs
from sklearn.impute import SimpleImputer
import numpy as np

# Missing_value como tiene que detectar los valores nulos
# strategy estrategia para remplazar los nulos. Mena es la media
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print(x)
