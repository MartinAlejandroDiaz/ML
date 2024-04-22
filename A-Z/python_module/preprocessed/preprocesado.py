# Plantilla de preprocesado

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# dataset = pd.read_csv('../Data.csv')
# x = dataset.iloc[:,:-1].values
# y = dataset.iloc[:,3].values


# # Tratamiento de los NAs
# from sklearn.impute import SimpleImputer
# import numpy as np

# # Missing_value como tiene que detectar los valores nulos
# # strategy estrategia para remplazar los nulos. Mena es la media
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer = imputer.fit(x[:,1:3])
# x[:,1:3] = imputer.transform(x[:,1:3])
# print(x)

# Codificar datos categoricos
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labellencoder_X = LabelEncoder()
# x[:,0]=labellencoder_X.fit_transform(x[:,0])
# print(x)

# print("//////////////////////////////")
# print("Transformamos una variable unica no comparable a una dummy")
# print("//////////////////////////////")
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# # Define el transformador de columnas
# # Especifica la(s) columna(s) que deseas codificar en one-hot
# transformador_columnas = ColumnTransformer(
#     transformers=[
#         ('onehot', OneHotEncoder(), [0])  # [0] es el índice de la columna que se va a codificar en one-hot
#     ],
#     remainder='passthrough'  # pasa las columnas no especificadas
# )

# # Ajusta y transforma tus datos usando el transformador de columnas
# X_codificado = transformador_columnas.fit_transform(x)
# print(X_codificado)
# print("//////////////////////////////")
# print("Codificado de respuestas")
# print("//////////////////////////////")
# labellencoder_y = LabelEncoder()
# y = labellencoder_y.fit_transform(y)
# print(y)

# print("//////////////////////////////")
# print("Codificado de respuestas")
# print("//////////////////////////////")

# # Dividir el data set en conjunto de entrenamiento y conjunto de testing
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# print(X_train, X_test, y_train, y_test)

# Escalado de variables
# x_stand = (x - mean(x)) / standarDeviation(x)
# x_norm = (x- min(x)) / (max(x)-min(x))

# pierde la pertenencia si no se usan las variables dummy en la estandarización
# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# X_train = sc_x.fit_transform(X_train)
# X_test = sc_x.transform(X_test)

def preprocessed(file, y_column, test_size, random_state=0):
    print("Preprocesado de datos")
    dataset = pd.read_csv(file)
    x = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,y_column].values
    print("Codificado de respuestas")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    print("Fin procesado")
    print(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test