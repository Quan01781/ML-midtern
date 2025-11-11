from sklearn.linear_model import LogisticRegression
from modelTraining import Model
import pandas as pd


class LogisticRegressionModel(Model):
    def __init__(self):
        # Inherit class Model
        super().__init__(LogisticRegression(max_iter=1000), "Logistic Regression")
    
    # Return DataFrame of features and their coefficients
    def feature_importance(self, feature_names):
        clf = self.model.named_steps['clf']  # lấy LogisticRegression bên trong pipeline
        ce = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": clf.coef_[0]
        }).sort_values("Coefficient", ascending=False)
        return ce

    # Return DataFrame of features and their coefficients
    # def feature_importance(self, feature_names):
    #     ce = pd.DataFrame({
    #         "Feature": feature_names,
    #         "Coefficient": self.model.coef_[0]
    #     }).sort_values("Coefficient", ascending=False)
    #     return ce