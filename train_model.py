from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    """
    Entrena un modelo de Random Forest con los datos de entrenamiento.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Modelo entrenado correctamente.")
    return model
