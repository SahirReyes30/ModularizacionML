from sklearn.metrics import accuracy_score

class EvaluateModel:
    """
    Clase para evaluar un modelo de Random Forest.
    """
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test

    def evaluate_model(model, X_test, y_test):
        """
        Evalúa el modelo usando el conjunto de prueba.
        """
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        return accuracy
