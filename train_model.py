from sklearn.ensemble import RandomForestClassifier

class TrainModel:
    """
    Clase para entrenar un modelo de Random Forest.
    """
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    
    def train_model(self, X_train, y_train):
        """
        Entrena un modelo de Random Forest con los datos de entrenamiento.
        """
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("Modelo entrenado correctamente.")
        return model

