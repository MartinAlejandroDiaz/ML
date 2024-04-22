def preprocessed(file, y_column, test_size, random_state=0):
    import pandas as pd
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

# from ..preprocessed.preprocesado import preprocessed
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def simple_linear_regression(file, y_column, test_size, random_state):
    X_train, X_test, y_train, y_test = preprocessed(file, y_column, test_size, random_state)
    if len(X_train) != len(y_train):
        return -1
    print("Entrenando el modelo")
    regression = LinearRegression()
    regression.fit(X_train, y_train)
    print("Predecir conjunto de entrenamiento")
    y_pred = regression.predict(X_test)
    print(y_pred)

    print("Recta de regresion lineal con el conjunto de entrenamiento")
    plt.scatter(X_train,y_train, color="red")
    plt.plot(X_train, regression.predict(X_train), color="blue")
    plt.title("Sueldo vs Años de Experiencia (Conjunto de entrenamiento)")
    plt.xlabel("Años de Experiencia")
    plt.ylabel("Sueldo (u$d)")
    plt.show()

    print("Recta de regresion lineal con el conjunto de testing")
    plt.scatter(X_test,y_test, color="red")
    plt.plot(X_train, regression.predict(X_train), color="blue")
    plt.title("Sueldo vs Años de Experiencia (Conjunto de entrenamiento)")
    plt.xlabel("Años de Experiencia")
    plt.ylabel("Sueldo (u$d)")
    plt.show()


simple_linear_regression('../data/Salary_Data.csv', 1, 1/3,0)
