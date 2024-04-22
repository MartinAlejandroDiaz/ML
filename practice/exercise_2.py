import pandas as pd
import numpy as np
import logging
import os
from ast import literal_eval

# Configuración de los registros
log_format_tipos_de_datos = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='tipos_de_datos.txt', level=logging.DEBUG, format=log_format_tipos_de_datos)

log_format_error_tipo = "%(asctime)s - %(levelname)s - %(message)s"
error_logger = logging.getLogger('error_tipo')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler('error_tipo.txt')
error_handler.setFormatter(logging.Formatter(log_format_error_tipo))
error_logger.addHandler(error_handler)

# Verificar si los archivos de registro existen, si no, crearlos
for file_name in ['tipos_de_datos.txt', 'error_tipo.txt']:
    if not os.path.exists(file_name):
        with open(file_name, 'w'):
            pass

# DataFrame original
data = {
    'A': [
        [{'ID': 'DOOR', 'VALUE': np.array([{'ID': 32424234, 'NAME': 5}])}, {'ID': 'HORSEPOWER', 'VALUE': np.array([{'ID': 32234344234, 'NAME': 5}])}],
        [{'ID': 'DOOR', 'VALUE': np.array([{'ID': 32424234, 'NAME': 5}])}, {'ID': 'HORSEPOWER', 'VALUE': np.array([{'ID': 32234344234, 'NAME': 5}])}]
    ]
}

df = pd.DataFrame(data)

# Función para convertir el string en una lista de diccionarios
def convertir_a_lista(string):
    return literal_eval(string)

# Aplicar la función a la columna 'A'
df['A'] = df['A'].apply(convertir_a_lista)

# Diccionario para almacenar los valores de cada ID como Series de pandas
valores = {}

# Iterar sobre cada fila del DataFrame
for index, row in df.iterrows():
    # Iterar sobre cada diccionario en la lista de la columna 'A'
    for diccionario in row['A']:
        id_ = diccionario['ID']
        valor = diccionario['VALUE']
        
        if id_ not in valores:
            valores[id_] = []
        
        valores[id_].append(valor)

# Diccionario para almacenar los formatos conocidos
formatos_conocidos = {
    'DOOR': int,
    'HORSEPOWER': str
}

# Iterar sobre los valores y crear las Series de pandas
for id_, lista_valores in valores.items():
    formato = formatos_conocidos.get(id_, object)
    valores[id_] = pd.Series([formato(valor) for sublist in lista_valores for valor in sublist], dtype=formato)
    
    # Validación de tipos y registro de logs
    for valor in lista_valores:
        if not isinstance(valor, formato):
            mensaje_error = f"Valor {valor} no coincide con el formato {formato} para ID '{id_}'"
            error_logger.error(mensaje_error)
        else:
            mensaje_info = f"Valor {valor} válido para ID '{id_}'"
            logging.debug(mensaje_info)

# Agregar las Series al DataFrame original
for id_, serie in valores.items():
    df[id_] = serie

print(df)
