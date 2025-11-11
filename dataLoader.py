import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.feature_names = None
        self.scaler = StandardScaler() # normalize data
    
    def process(self):
        self.data = pd.read_csv(self.filepath)
        # print("Missing values:\n", self.data.isnull().sum())
        # # One-Hot Encoding
        # data = pd.get_dummies(data, drop_first=True)

        X = self.data.drop("target", axis=1)
        y = self.data["target"]
        
        # Save columns names
        self.feature_names = X.columns
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)


        
        return X_train_scaled, X_test_scaled, y_train, y_test

