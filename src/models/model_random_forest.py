#dictionnaire exhaustif des hyperparamètres

# src/models/random_forest.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import os

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialisation du modèle Random Forest avec des hyperparamètres de base.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.name = "RandomForest"

    def train(self, X_train, y_train):
        """
        Entraîne le modèle Random Forest sur les données fournies.
        """
        self.model.fit(X_train, y_train)
        print(f"entrainement du modèle effectuée:{self.model}")

    def predict(self, X):
        """
        Prédiction avec le modèle Random Forest.
        """
        print(f"prédiction du modèle effectuée :{self.model}")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Prédiction avec le modèle Random Forest.
        """
        print(f"prédiction proba du modèle effectuée :{self.model}")
        return self.model.predict_proba(X)        

    def tune_parameters(self, X, y, param_grid, cv=3):
        """
        Réglage des hyperparamètres du modèle Random Forest avec GridSearchCV.
        """
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
        print(f"Meilleur score de validation croisée : {grid_search.best_score_:.4f}")

    def save_model(self, path='model.joblib'):
        """
        Sauvegarde le modèle entraîné sur le disque.
        """
        joblib.dump(self.model, path)
        print(f"Modèle sauvegardé à l'emplacement : {path}")

    def load_model(self, path='model.joblib'):
        """
        Charge un modèle depuis le disque.
        """
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Modèle chargé depuis : {path}")
        else:
            print("Le chemin du modèle spécifié n'existe pas.")
