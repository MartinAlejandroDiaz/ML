import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

# plt.scatter(X, y, color="red")
# plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color="blue")
# plt.title("Modelo de Regresión Polinomica")
# plt.ylabel("Sueldo en ($)")
# plt.show()

# plt.scatter(X, y, color="red")
# plt.plot(X, lin_reg.predict(X), color="blue")
# plt.title("Modelo de Regresión Polinomica")
# plt.ylabel("Sueldo en ($)")
# plt.show()

print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))