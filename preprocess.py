import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data(data, dataType, target_column=None):
    """
    Preprocesa los datos según el tipo de dataset seleccionado.
    """
    if dataType == "iris":
        # Asignar nombres a las columnas
        data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
        X = data.drop(columns=["target"])
        y = data["target"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

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
