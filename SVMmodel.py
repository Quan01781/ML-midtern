from sklearn.svm import SVC
from modelTraining import Model


class SVMModel(Model):
    def __init__(self):
        # Inherit class Model
        super().__init__(SVC(probability=True), "SVM")