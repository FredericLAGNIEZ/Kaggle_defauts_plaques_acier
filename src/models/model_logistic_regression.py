#dictionnaire exhaustif des hyperparamètres
# src/models/logictic_regression.py
# utiliser la regression logistique comme une baseline pour la prédiction des autres modles
# s'assurer de l'absence de variables corrélées entre elles --> features selection, et réduire les outliers

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

class LogisticRegressionModel:

    def __init__(self,penalty='l1',C=0.1, max_iter=100, solver = 'saga',random_state=42):
        """
        Initialisation du modèle de Regression Logistique avec des hyperparamètres de base.
        """
        valid_solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
        if solver not in valid_solvers:
            raise ValueError(f"La valeur de solver doit être dans {valid_solvers}")
        #initiation des attributs
        self.model = LogisticRegression(penalty=penalty,C=C, max_iter=max_iter, solver = solver,random_state=random_state)
        self.name = "LogisticRegression"

    def train(self, X_train, y_train):
        """
        Entraîne le modèle Regression Logistique sur les données fournies.
        """
        self.model.fit(X_train, y_train)
        print(f"entrainement du modèle {self.model} effectué.")

    def predict(self, X):
        """
        Prédiction avec le modèle Regression Logistique .
        """
        print(f"Prédiction du modèle {self.model} effectué.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Prédiction des probabilité avec le modèle de Regression Logistique. 
        """
        return self.model.predict_proba(X)        


    def tune_parameters(self, X, y, param_grid, cv=3):
        """
        Réglage des hyperparamètres du modèle de Regression Logistique avec GridSearchCV.
        Ceci n'est pas indispensable avec ce modèle.
        """
        grid_search = GridSearchCV(self.model,param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_

        print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
        print(f"Meilleur score de validation croisée : {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def score(self, X_test, y_test):
        """
        Calcule le score du modèle sur les données de test.
        """
        return self.model.accuracy_score(X_test, y_test)

    def save_model(self, path='model.joblib'):
        """
        Sauvegarde le modèle entraîné sur le disque (chemin path).
        """
        joblib.dump(self.model,path)
        print(f"Modèle sauvegardé à l'emplacement : {path}")

    def load_model(self, path='model.joblib'):
        """
        Charge un modèle depuis le disque (chemin path).
        """
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Modèle chargé depuis : {path}")
        else:
            print("Le chemin du modèle spécifié n'existe pas.")

    def __str__(self):
        """
        Renvoie une représentation sous forme de chaîne de l'objet LogisticRegression.
        """
        return f"{self.name}(penalty={self.model.penalty}, C={self.model.C}, max_iter={self.model.max_iter}, solver={self.model.solver}, random_state={self.model.random_state})"


    def multioutput_pipeline(self,preprocessor):
        """
        Crée un pipeline qui combine un prétraitement (preprocessor) et un modèle de régression multi-sortie
        """
    # Créer un pipeline avec un prétraitement (ex: normalisation) et un modèle de régression multi-target
        pipeline = Pipeline([
        ('preprocessor', preprocessor),    
        ('regressor', MultiOutputRegressor(self.model))
    ])
        return pipeline

