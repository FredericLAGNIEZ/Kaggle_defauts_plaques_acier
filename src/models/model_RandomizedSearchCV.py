from sklearn.model_selection import RandomizedSearchCV
import joblib

class RandomSearch:
    
    def __init__(self, estimator,param_grid,n_iter=10, cv=5, scoring='accuracy', verbose=1, n_jobs=-1):
        """
        Initialisation avec des arguments dynamiques.

        :param estimator: Le modèle/estimateur sur lequel effectuer la recherche par grille.
        :param param_grid: Grille de paramètres à tester pour l'estimateur.
        :param cv: Nombre de plis pour la validation croisée.
        :param scoring: Métrique de scoring à utiliser.
        :param verbose: Niveau de verbosité de la sortie.
        :param n_jobs: Nombre de jobs à exécuter en parallèle.
        """
        self.random_search = RandomizedSearchCV(estimator,param_grid,n_iter=n_iter,  cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs)

    def fit(self, X, y):
        """
        Exécute la recherche par grille sur les données fournies.
        
        :param X: Les caractéristiques d'entraînement.
        :param y: Les étiquettes cibles.
        """
        
        self.random_search.fit(X, y)
        print(f"Meilleurs paramètres: {self.random_search.best_params_}")
        print(f"Meilleur score: {self.random_search.best_score_}")

    def predict(self, X):
        """
        Fait des prédictions avec le meilleur modèle trouvé.
        
        :param X: Les caractéristiques pour lesquelles faire des prédictions.
        :return: Les prédictions du modèle.
        """
        if self.random_search is None:
            raise Exception("RandomizedSearchCV n'a pas encore été ajusté.")
        return self.random_search.predict(X)

    def save_best_model(self, file_path):
        """
        Sauvegarde le meilleur modèle trouvé sur le disque.
        
        :param file_path: Chemin du fichier où sauvegarder le modèle.
        """
        if self.random_search is None:
            raise Exception("RandomizedSearchCV n'a pas encore été ajusté.")
        joblib.dump(self.random_search.best_estimator_, file_path)
