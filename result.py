import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay



class RpEvaluation:
    def __init__(self, models):
        self.models = models
        
    # Show precision, recall, f1-score
    def get_evaluation(self):
        print(f"{self.models.name}:")
        print(f"  Test Accuracy = {self.models.accuracy:.2f}")
        
        for avg in ['macro', 'weighted']:
            print(f"  {avg} average:")
            print(f"    Precision = {self.models.precision[avg]:.2f}")
            print(f"    Recall    = {self.models.recall[avg]:.2f}")
            print(f"    F1-score  = {self.models.f1[avg]:.2f}")

        print(f"  CV Accuracy = {self.models.cross_vali:.2f}")
        print(f"  Confusion Matrix:\n{self.models.confusion}\n")
        if self.models.roc_auc:
            print(f"  ROC-AUC = {self.models.roc_auc:.2f}\n")

class Result:
    def __init__(self, models):
        self.models = models
       
    def show_evaluation(self):
        print("---Model Evaluation---")
        for model in self.models:
            RpEvaluation(model).get_evaluation()
        
    # Draw accuracy and cross validiation
    def draw_results(self):
        names = [model.name for model in self.models]
        accuracy = [model.accuracy for model in self.models]
        cross_val = [model.cross_vali for model in self.models]

        x = np.arange(len(names))
        w = 0.35

        plt.figure(figsize=(8,5))
        plt.bar(x - w/4, accuracy, width=w, label="Test Accuracy", color='skyblue', align='center')
        # bars = plt.bar(x, accuracy, label="Test Accuracy", color='skyblue', align='center')
        # bars = plt.bar(x, cross_val, label="CV Accuracy", color='salmon', align='center')
        plt.bar(x + w/4, cross_val, width=w, label="CV Accuracy", color='salmon', align='edge')
        # for bar in bars:
        #     height = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
        #             f"{height:.2f}", ha='center', va='bottom')

        plt.xticks(x, names)
        plt.ylabel("Accuracy")
        plt.ylim(0,1)
        plt.title("Comparison of Models")
        plt.legend()
        # plt.savefig("overall.png", dpi=300, bbox_inches="tight")  
        plt.show(block=False)

    # Draw 5 important features
    def plot_feature_importance(self, lr_features, rf_features):
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        # Plot Logistic Regression
        bars_lr = axes[0].barh(lr_features["Feature"].head(5)[::-1], # get 5 most important attributes
                    lr_features["Coefficient"].head(5)[::-1], color='steelblue')
        for bar in bars_lr:
            width = bar.get_width()
            axes[0].text(width, bar.get_y() + bar.get_height()/2,
                        f"{width:.2f}", ha='left', va='center')

        axes[0].set_title("Top 5 Features (Logistic Regression)")
        axes[0].set_xlabel("Coefficient")

        # Plot Random Forest
        bars_rf = axes[1].barh(rf_features["Feature"].head(5)[::-1], # get 5 most important attributes
                    rf_features["Importance"].head(5)[::-1], color='seagreen')  
        for bar in bars_rf:
            width = bar.get_width()
            axes[1].text(width, bar.get_y() + bar.get_height()/2,
                        f"{width:.2f}", ha='left', va='center')
     
        axes[1].set_title("Top 5 Features (Random Forest)")
        axes[1].set_xlabel("Feature Importance")

        plt.tight_layout()
        # plt.savefig("5.png", dpi=300, bbox_inches="tight")  
        plt.show(block=False)
    
    # Confusion matrix 
    def plot_confusion_matrices(self, X_test, y_test):
        fig, axes = plt.subplots(2, 2, figsize=(10,8))
        axes = axes.ravel() # Change to 1D array

        for i, model in enumerate(self.models):
            ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test), ax=axes[i], cmap='Blues', colorbar=False
            )
            axes[i].set_title(model.name)
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")

        plt.tight_layout()
        # plt.savefig("matrix.png", dpi=300, bbox_inches="tight")  
        plt.show(block=False)

    # ROC_Curve Plot
    def plot_roc_curves(self, X_test, y_test):
        fig, axes = plt.subplots(2, 2, figsize=(10,8))
        axes = axes.ravel()  # Change to 1D array

        for i, model in enumerate(self.models):
            RocCurveDisplay.from_estimator(model.model, X_test, y_test, ax=axes[i])
            axes[i].plot([0,1],[0,1],'--', color='gray', label="Random guess")
            axes[i].set_title(f"{model.name} - ROC Curve")
            axes[i].legend(loc="lower right")

        plt.tight_layout()
        # plt.savefig("AUC.png", dpi=300, bbox_inches="tight")  
        plt.show()