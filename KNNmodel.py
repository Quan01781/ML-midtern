from sklearn.neighbors import KNeighborsClassifier
from modelTraining import Model


class KNNModel(Model):
    def __init__(self):
        # Inherit class Model
        super().__init__(KNeighborsClassifier(n_neighbors=5), "KNN")