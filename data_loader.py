import pandas as pd

class DataLoader:
    """
    Clase para cargar datasets desde URLs.
    """
    
    def __init__(self):
        """
        Inicializa las URLs y nombres de los datasets disponibles.
        """
        self.datasets = {
            "1": {
                "name": "iris",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                "header": None
            },
            "2": {
                "name": "wine",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
                "header": None
            },
            "3": {
                "name": "breast_cancer",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                "header": None
            },
            "4": {
                "name": "mall",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv",
                "header": 0
            },
            "5": {
                "name": "mnist",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
                "header": None
            }
        }

    def select_data(self):
        """
        Solicita al usuario seleccionar un dataset y lo carga desde la URL correspondiente.
        
        Returns:
            tuple: Un DataFrame con los datos cargados y el nombre del dataset.
        """
        while True:
            print("Selecciona el dataset que deseas cargar:")
            for key, dataset in self.datasets.items():
                print(f"{key}. {dataset['name'].capitalize()}")
            
            choice = input("Introduce el número correspondiente al dataset: ").strip()
            
            if choice in self.datasets:
                selected = self.datasets[choice]
                print(f"Cargando dataset {selected['name'].capitalize()}...")
                data = pd.read_csv(selected["url"], header=selected["header"])
                return data, selected["name"]
            else:
                print("Opción no válida. Por favor, intenta de nuevo.")
