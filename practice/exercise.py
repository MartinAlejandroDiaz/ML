import pandas as pd
from ast import literal_eval
import logging
import os

# Configurar el registro de tipos_de_datos
log_format_tipos_de_datos = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='tipos_de_datos.txt', level=logging.DEBUG, format=log_format_tipos_de_datos)

# Configurar el registro de error_tipo
log_format_error_tipo = "%(asctime)s - %(levelname)s - %(message)s"
error_logger = logging.getLogger('error_tipo')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler('error_tipo.txt')
error_handler.setFormatter(logging.Formatter(log_format_error_tipo))
error_logger.addHandler(error_handler)

# Verificar si el archivo de registro existe para tipos_de_datos, si no, crearlo
if not os.path.exists('tipos_de_datos.txt'):
    with open('tipos_de_datos.txt', 'w'):
        pass

# Verificar si el archivo de registro existe para error_tipo, si no, crearlo
if not os.path.exists('error_tipo.txt'):
    with open('error_tipo.txt', 'w'):
        pass

# Tu DataFrame original
data = {
    'A': [
        "[{'ID': 'DOOR', 'VALUE': 4}, {'ID': 'HORSEPOWER', 'VALUE': 4}, {'ID': 'BRAND', 'VALUE': 'SUV'}]",
        "[{'ID': 'DOOR', 'VALUE': 4}, {'ID': 'HORSEPOWER', 'VALUE': 4}]",
        "[{'ID': 'DOOR', 'VALUE': '2022-01-02'}, {'ID': 'BRAND', 'VALUE': 'SUV'}]"
    ]
}

df = pd.DataFrame(data)

# Función para convertir el string en una lista de diccionarios
def convertir_a_lista(string):
    return literal_eval(string)

# Aplicar la función a la columna 'A'
df['A'] = df['A'].apply(convertir_a_lista)

# Crear un diccionario para almacenar los formatos conocidos
formatos_conocidos = {
    'DOOR': int,  # 'DOOR' siempre es un número
    'HORSEPOWER': str,  # 'HORSEPOWER' es una cadena de texto
    'MODEL': str  # 'MODEL' es una cadena de texto
}

# Crear un diccionario para almacenar los valores de cada ID como Series de pandas
valores = {}

# Iterar sobre cada fila del DataFrame
for index, row in df.iterrows():
    # Iterar sobre cada diccionario en la lista de la columna 'A'
    for diccionario in row['A']:
        # Obtener el ID y el valor del diccionario
        id_ = diccionario['ID']
        valor = diccionario['VALUE']
        # Verificar si el formato es conocido para el ID
        if id_ in formatos_conocidos:
            formato = formatos_conocidos[id_]
        else:
            # Si el ID es desconocido, registrar un mensaje de depuración en tipos_de_datos.txt
            mensaje = f"Formato desconocido para ID '{id_}': {valor}"
            logging.debug(mensaje)
            formato = object  # Se establece un formato predeterminado como objeto
        # Aplicar la estrategia de limpieza adecuada según el tipo de dato
        # Aplicar la estrategia de limpieza adecuada según el tipo de dato
        if formato == int:
            # Si es un número, calcular la media
            try:
                if valor is not None:
                    valor = float(valor)
                else:
                    valor = None
            except ValueError:
                mensaje_error = f"Valor no numérico en el índice {index}: {valor}"
                error_logger.error(mensaje_error)
                valor = None

        elif formato == str:
            # Si es un enum de palabras, calcular la moda
            pass  # Aquí puedes implementar el cálculo de la moda
        elif formato == object:
            # Si es una fecha, verificar si corresponde al formato de la mayoría
            pass  # Aquí puedes implementar la verificación del formato de fecha
        # Si el ID ya está en el diccionario de valores, agregar el valor a la Serie
        if id_ in valores:
            valores[id_].loc[index] = formato(valor)
        # Si no está en el diccionario de valores, crear una nueva Serie con el valor
        else:
            valores[id_] = pd.Series(index=df.index, dtype=formato)
            valores[id_].loc[index] = formato(valor)

# Agregar las Series al DataFrame original
for id_, serie in valores.items():
    df[id_] = serie

print(df)
