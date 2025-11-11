from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay


class Model:
    def __init__(self, models, name):
        self.model = models
        self.name = name
        self.accuracy = None
        self.confusion = None
        self.cross_vali = None
        self.roc_auc = None
        self.precision = {}
        self.recall = {}
        self.f1 = {}
        
    # Training
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # Predict
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    # get macro, weighted average, roc_auc score, probability score
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.confusion = confusion_matrix(y_test, y_pred)
        y_proba = self.model.predict_proba(X_test)[:,1]
        self.roc_auc = roc_auc_score(y_test, y_proba)

        for avg in ['macro', 'weighted']:
            self.precision[avg] = precision_score(y_test, y_pred, average=avg)
            self.recall[avg] = recall_score(y_test, y_pred, average=avg)
            self.f1[avg] = f1_score(y_test, y_pred, average=avg)

    #cross validiation
    def cross_val(self, X, y, cv=5):
        scores = cross_val_score(self.model, X, y, cv=cv)
        self.cross_vali = scores.mean()


    # def cross_val(self, X, y, cv=5):
    #     from sklearn.model_selection import StratifiedKFold
    #     from sklearn.preprocessing import StandardScaler
    #     import numpy as np

    #     skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    #     scores = []

    #     for train_idx, test_idx in skf.split(X, y):
    #         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    #         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    #         scaler = StandardScaler()
    #         X_train_scaled = scaler.fit_transform(X_train)
    #         X_test_scaled = scaler.transform(X_test)

    #         self.model.fit(X_train_scaled, y_train)
    #         score = self.model.score(X_test_scaled, y_test)
    #         scores.append(score)

    #     self.cross_vali = np.mean(scores)