import datetime

data = {
    'A': [
        [{'ID': 'DOOR', 'VALUES': [{'ID': 2341241234, 'NAME': 4}] }, {'ID': 'HORSEPOWER', 'VALUES': [{'ID': 2341241234, 'NAME': 4}] }, {'ID': 'BRAND', 'VALUES': 'SUV'}],
        [{'ID': 'DOOR', 'VALUES': [{'ID': 2341241234, 'NAME': 4}] }, {'ID': 'HORSEPOWER', 'VALUES': [{'ID': 2341241234, 'NAME': 4}] }],
        [{'ID': 'DOOR', 'VALUES': [{'ID': 2341241234, 'NAME':'2022-01-02'}]}, {'ID': 'BRAND', 'VALUES': 'SUV'}]
    ]
}
# (Tipo de suspensión)
suspension_type = ['independiente', 'barra de torsión', 'de doble horquilla']
# kilometros
KILOMETERS = float
# MARca
brand = ['Toyota', 'Ford', 'Honda']
# (Fecha de fabricación
manufacture_date = Date
# model
model = ['Camry', 'Mustang', 'Civic']
# Tipo de motor
engine_type = ['gasolina', 'diésel', 'híbrido', 'eléctrico']
# Desplazamiento del motor
engine_displacement = float
# Caballos de fuerza
horsepower = float
# Eficiencia de combustible
fuel_efficiency = float
# Tipo de transmisión
transmission_type = ['manual', 'automática', 'continuamente variable']
# door
door = [2, 3, 4, 5]
# Capacidad de asientos
seating_capacity = int
# Calificación de seguridad
safety_rating = ['accidente', 'nuevo']
# color
color = ['rojo', 'azul', 'negro']
# Tamaño de neumáticos
tire_size = int
# Peso
weight = float
# precio
price = float



import pandas as pd
from datetime import date
import random

# Define la cantidad de registros que deseas
num_records = 5

# Define los valores para cada campo
data = {
    'A': [{'ID': 'DOOR', 'VALUE': 4}] * num_records
}

# Crea el DataFrame
df = pd.DataFrame(data)

# Función para obtener el valor de un ID específico de la lista de diccionarios
def get_value_by_id(id_, lista):
    for diccionario in lista:
        if diccionario['ID'] == id_:
            return diccionario['VALUE']
    return None  # Si no se encuentra el ID, devuelve None

# Define los valores para cada columna basados en la función get_value_by_id
suspension_type = ['independiente'] * num_records
KILOMETERS = [random.randint(10000, 50000) for _ in range(num_records)]
brand = ['Toyota'] * num_records
manufacture_date = [date(2022, 4, 15)] * num_records
model = ['Camry'] * num_records
engine_type = ['gasolina'] * num_records
engine_displacement = [random.uniform(1.5, 3.5) for _ in range(num_records)]
horsepower = [random.randint(150, 300) for _ in range(num_records)]
fuel_efficiency = [random.uniform(20, 40) for _ in range(num_records)]
transmission_type = ['manual'] * num_records
door = [get_value_by_id('DOOR', df['A']) for _ in range(num_records)]
seating_capacity = [random.randint(2, 7) for _ in range(num_records)]
safety_rating = [None] * num_records
color = ['rojo', 'azul', 'negro', 'verde', 'blanco']
tire_size = [random.randint(15, 20) for _ in range(num_records)]
weight = [random.randint(2000, 4000) for _ in range(num_records)]
price = [random.randint(20000, 40000) for _ in range(num_records)]

# Crea un diccionario con los valores de cada campo
data = {
    'suspension_type': suspension_type,
    'kilometers': KILOMETERS,
    'brand': brand,
    'manufacture_Date': manufacture_date,
    'model': model,
    'engine_type': engine_type,
    'engine_displacement': engine_displacement,
    'horsepower': horsepower,
    'fuel_efficiency': fuel_efficiency,
    'transmission_type': transmission_type,
    'door': door,
    'seating_capacity': seating_capacity,
    'safety_rating': safety_rating,
    'color': [random.choice(color) for _ in range(num_records)],
    'tire_size': tire_size,
    'weight': weight,
    'price': price
}

# Crea el DataFrame
df = pd.DataFrame(data)

# Imprime el DataFrame
print(df)
