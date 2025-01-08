import pandas as pd

def select_data():
    """
    Función para cargar diferentes datasets desde URLs basado en la entrada del usuario.
    """
    print("Selecciona el dataset que deseas cargar:")
    print("1. Iris")
    print("2. Wine")
    print("3. Breast Cancer")
    print("4. Mall Customers")
    print("5. MNIST Digits")
    
    choice = input("Introduce el número correspondiente al dataset: ").strip()

    if choice == "1":
        print("Cargando dataset Iris...")
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None), "iris"
    elif choice == "2":
        print("Cargando dataset Wine...")
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None), "wine"
    elif choice == "3":
        print("Cargando dataset Breast Cancer...")
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", header=None), "breast_cancer"
    elif choice == "4":
        print("Cargando dataset Mall Customers...")
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"), "mall"
    elif choice == "5":
        print("Cargando dataset MNIST Digits...")
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None), "mnist"
    else:
        print("Opción no válida. Por favor, intenta de nuevo.")
        return select_data()  # Volver a pedir la entrada

