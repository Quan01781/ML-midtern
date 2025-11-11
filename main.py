from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from KNNmodel import KNNModel
from LRmodel import LogisticRegressionModel
from RFmodel import RandomForestModel
from SVMmodel import SVMModel
from dataLoader import DataLoader
from modelTraining import Model
from result import Result


class Main:
    def __init__(self, path):
        self.path = path
        self.models = [
            LogisticRegressionModel(),
            SVMModel(),
            KNNModel(),
            RandomForestModel()]
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_scaled = None
        self.y = None

    # Preprocess data 
    def load_data(self):
        self.data = DataLoader(self.path)
        self.X_train, self.X_test, self.y_train, self.y_test = self.data.process()


    # Train and evaluate each model
    def train_and_evaluate(self):
        for model in self.models:
            model.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model.model)
            ])

            model.train(self.X_train, self.y_train)
            model.evaluate(self.X_test, self.y_test)
            model.cross_val(self.X_train, self.y_train)

    # Get feature importance for LR & RF 
    def get_feature_importance(self):
        lr_model = next(model for model in self.models if model.name == "Logistic Regression")
        rf_model = next(model for model in self.models if model.name == "Random Forest")
        lr_feat = lr_model.feature_importance(self.data.feature_names)
        rf_feat = rf_model.feature_importance(self.data.feature_names)
        return lr_feat, rf_feat

    # Draw all results
    def get_results(self, lr_feat, rf_feat):
        result = Result(self.models)
        result.show_evaluation()
        result.draw_results()
        result.plot_feature_importance(lr_feat, rf_feat)
        result.plot_confusion_matrices(self.X_test, self.y_test)
        result.plot_roc_curves(self.X_test, self.y_test)

    #  Main runner 
    def run(self):
        self.load_data()
        self.train_and_evaluate()
        lr_feat, rf_feat = self.get_feature_importance()
        self.get_results(lr_feat, rf_feat)


if __name__ == "__main__":
    main = Main("C:/Machina Learn/machina learn midtern/heart_disease.csv")
    main.run()