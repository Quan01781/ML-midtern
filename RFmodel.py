from sklearn.ensemble import RandomForestClassifier
from modelTraining import Model
import pandas as pd


class RandomForestModel(Model):
    def __init__(self):
        # Inherit class Model
        super().__init__(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")

    # Return DataFrame of features
    def feature_importance(self, feature_names):
        clf = self.model.named_steps['clf']  # lấy RandomForest bên trong pipeline
        imF = pd.DataFrame({
            "Feature": feature_names,
            "Importance": clf.feature_importances_
        }).sort_values("Importance", ascending=False)
        return imF
    
    # Return DataFrame of features
    # def feature_importance(self, feature_names):
    #     imF = pd.DataFrame({
    #         "Feature": feature_names,
    #         "Importance": self.model.feature_importances_
    #     }).sort_values("Importance", ascending=False)
    #     return imF 

