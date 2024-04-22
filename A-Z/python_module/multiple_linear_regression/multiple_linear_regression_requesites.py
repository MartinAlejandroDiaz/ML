# ///////////////////////////////////////////////////////////////////////////////
# Linealidad
# ///////////////////////////////////////////////////////////////////////////////
import numpy as np

def check_linearity(points):
    points = np.array(points)    
    diffs = np.diff(points, axis=0)
    if np.all(np.isclose(diffs / diffs[0], diffs[0] / np.linalg.norm(diffs[0]))):
        return True
    else:
        return False
points = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

if check_linearity(points):
    print("La curva es lineal en el espacio.")
else:
    print("La curva no es lineal en el espacio.")


# ///////////////////////////////////////////////////////////////////////////////
# Homocedasticidad
# ///////////////////////////////////////////////////////////////////////////////

# Generar datos de ejemplo para una regresión lineal
def generate_random_values():
    np.random.seed(0)
    X = np.random.rand(100, 1) * 10
    return X, 2 * X + np.random.randn(100, 1) * 2

from sklearn.linear_model import LinearRegression
class Homoscedasticity:
    def train_model_homoscedasticity(X,y):
        print("Entrenar el modelo de regresión lineal")
        model = LinearRegression()
        model.fit(X, y)
        return model

    def get_homoscedasticity_residuals(model: LinearRegression, X):
        print("Calcular los residuos")
        return y - model.predict(X)

    def graph_homoscedasticity(model, X, residuals):
        import matplotlib.pyplot as plt
        print("Graficar los residuos vs. valores predichos")
        plt.scatter(model.predict(X), residuals)
        plt.xlabel("Valores Predichos")
        plt.ylabel("Residuos")
        plt.title("Gráfico de Residuos vs. Valores Predichos")
        plt.axhline(y=0, color='r', linestyle='-')
        plt.show()

    def check_null_homoscedasticity(max_p_val, y, x):
        import statsmodels
        _, p_value, _, _ = statsmodels.stats.diagnostic.het_breuschpagan(y, np.column_stack((np.ones_like(x), x)))
        print("p-valor de la prueba de Breusch-Pagan:", p_value)
        return max_p_val > p_value

# ///////////////////////////////////////////////////////////////////////////////
# Normalidad multivariable
# ///////////////////////////////////////////////////////////////////////////////

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generar datos de ejemplo con distribución normal multivariable
np.random.seed(0)
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
data = np.random.multivariate_normal(mean, cov, 1000)

# Gráfico de Q-Q para cada variable
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for i in range(2):
    stats.probplot(data[:, i], dist="norm", plot=axs[i])
    axs[i].set_title(f'Variable {i+1}')
plt.tight_layout()
plt.show()

# Prueba de Shapiro-Wilk para cada variable
for i in range(2):
    _, p_value = stats.shapiro(data[:, i])
    print(f'Variable {i+1}: p-valor de Shapiro-Wilk = {p_value}')

# ///////////////////////////////////////////////////////////////////////////////
# Independencia de los errores
# ///////////////////////////////////////////////////////////////////////////////

import numpy as np
import statsmodels.api as sm

# Generar datos de ejemplo para una regresión lineal
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + np.random.randn(100, 1) * 2  # Modelo con error independiente

# Ajustar el modelo de regresión lineal
X = sm.add_constant(X)  # Agregar un intercepto a la matriz de características
model = sm.OLS(y, X)
results = model.fit()

# Calcular los residuos
residuals = results.resid

# Prueba de Durbin-Watson
dw_test_statistic = sm.stats.stattools.durbin_watson(residuals)
print("Estadístico de Durbin-Watson:", dw_test_statistic)


# ///////////////////////////////////////////////////////////////////////////////
# Ausencia de multicolinealidad
# ///////////////////////////////////////////////////////////////////////////////


import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar el conjunto de datos de Boston Housing
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Calcular el factor de inflación de la varianza (VIF) para cada variable independiente
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)
