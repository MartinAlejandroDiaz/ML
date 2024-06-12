import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
"""

from sklearn.svm import SVR

regression = SVR(kernel="rbf")

y_pred = regression.predict([[6.5]])
print(y_pred)