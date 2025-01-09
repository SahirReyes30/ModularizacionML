import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Preprocess:
    """
    Clase para preprocesar los datos según el tipo de dataset seleccionado.
    """

    def __init__(self):
        """
        Inicializa el preprocesador
        """
        self.scaler = StandardScaler()

    def preprocess_data(self, data, dataType, target_column=None):
        """
        Preprocesa los datos según el tipo de dataset seleccionado.

        Args:
            data (DataFrame): El DataFrame con los datos a preprocesar.
            dataType (str): El tipo de dataset seleccionado.
            target_column (str, opcional): El nombre de la columna objetivo.

        Returns:
            tuple: Un tuple con los conjuntos de entrenamiento y prueba.
        """
        if dataType == "iris":
            return self.preprocess_iris(data)
            

        elif dataType == "wine":
            # Asignar nombres a las columnas
            data.columns = ["target"] + [f"feature_{i}" for i in range(1, 14)]
            X = data.drop(columns=["target"])
            y = data["target"]
            return train_test_split(X, y, test_size=0.2, random_state=42)

        elif dataType == "breast_cancer":
            # Asignar nombres a las columnas y manejar valores faltantes
            data.columns = ["id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
                            "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
                            "bland_chromatin", "normal_nucleoli", "mitoses", "target"]
            data = data.replace("?", pd.NA).dropna().drop(columns=["id"])
            X = data.drop(columns=["target"])
            y = data["target"].astype(int)
            return train_test_split(X, y, test_size=0.2, random_state=42)

        elif dataType == "mall":
            # No hay etiquetas en este dataset
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            return data_scaled

        elif dataType == "mnist":
            # Separar características y etiquetas
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            X = X / 255.0  # Normalización
            return train_test_split(X, y, test_size=0.2, random_state=42)

        else:
            raise ValueError("Tipo de datos no soportado.")
        

    def _preprocess_iris(self, data):
        """
        Preprocesa los datos del dataset Iris.

        Args:
            data (DataFrame): El DataFrame con los datos a preprocesar.

        Returns:
            tuple: Un tuple con los conjuntos de entrenamiento y prueba.
        """
        # Asignar nombres a las columnas
        data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
        X = data.drop(columns=["target"])
        y = data["target"]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def _preprocess_wine(self, data):
        """
        Preprocesa los datos del dataset Wine.

        Args:
            data (DataFrame): El DataFrame con los datos a preprocesar.

        Returns:
            tuple: Un tuple con los conjuntos de entrenamiento y prueba.
        """
        # Asignar nombres a las columnas
        data.columns = ["target"] + [f"feature_{i}" for i in range(1, 14)]
        X = data.drop(columns=["target"])
        y = data["target"]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def _preprocess_breast_cancer(self, data):
        """
        Preprocesa los datos del dataset Breast Cancer.

        Args:
            data (DataFrame): El DataFrame con los datos a preprocesar.

        Returns:
            tuple: Un tuple con los conjuntos de entrenamiento y prueba.
        """
        # Asignar nombres a las columnas y manejar valores faltantes
        data.columns = ["id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
                        "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
                        "bland_chromatin", "normal_nucleoli", "mitoses", "target"]
        data = data.replace("?", pd.NA).dropna().drop(columns=["id"])
        X = data.drop(columns=["target"])
        y = data["target"].astype(int)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def _preprocess_mall(self, data):
        """
        Preprocesa los datos del dataset Mall Customers.

        Args:
            data (DataFrame): El DataFrame con los datos a preprocesar.

        Returns:
            array: Un array con los datos preprocesados.
        """
        # No hay etiquetas en este dataset
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        return data_scaled
    
    def _preprocess_mnist(self, data):
        """
        Preprocesa los datos del dataset MNIST.

        Args:
            data (DataFrame): El DataFrame con los datos a preprocesar.

        Returns:
            tuple: Un tuple con los conjuntos de entrenamiento y prueba.
        """
        # Separar características y etiquetas
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X = X / 255.0  # Normalización
