from data_loader import select_data
from preprocess import preprocess_data
from train_model import train_model
from evaluate import evaluate_model

def main():

    # Paso 1: Cargar datos
    raw_data, dataType = select_data()

    # Paso 2: Preprocesar datos
    if dataType in ["iris", "wine", "breast_cancer", "mnist"]:
        X_train, X_test, y_train, y_test = preprocess_data(raw_data, dataType)
        print(f"Dataset {dataType} procesado correctamente para modelos supervisados.")
    elif dataType == "mall":
        processed_data = preprocess_data(raw_data, dataType)
        print("Dataset Mall Customers procesado correctamente para clustering (KMeans).")

    # Paso 3: Entrenar el modelo
    model = train_model(X_train, y_train)

    # Paso 4: Evaluar el modelo
    accuracy = evaluate_model(model, X_test, y_test)

    print(f"Accuracy del modelo: {accuracy:.2f}")

if __name__ == "__main__":
    main()
