from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo usando el conjunto de prueba.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy
