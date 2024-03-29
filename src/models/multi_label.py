import numpy as np

class MultiLabelModelWrapper:
    """
    Un wrapper pour des modèles d'apprentissage automatique qui permet de traiter des tâches multi-labels
    en adaptant la sortie du modèle aux exigences spécifiques du scoring multi-label.
    """
    def __init__(self, model):
        """
        Initialise l'enveloppeur avec un modèle d'apprentissage automatique.
        
        :param model: Un modèle d'apprentissage automatique de scikit-learn ou compatible.
        """
        self.model = model
        
    def fit(self, X, y):
        """
        Entraîne le modèle sur les données fournies.
        
        :param X: Les caractéristiques d'entrainement.
        :param y: Les étiquettes cibles d'entrainement.
        :return: self, l'instance du modèle après l'entraînement.
        """
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """
        Prédit les probabilités des classes pour chaque sortie dans une tâche multi-label.
        
        :param X: Les caractéristiques pour lesquelles prédire les probabilités.
        :return: Un array numpy des probabilités prédites pour chaque classe et chaque sortie.
        """
        pred_list = self.model.predict_proba(X)
        pred = np.hstack([pred[:, 1].reshape(-1, 1) for pred in pred_list])
        return pred
    
    def get_params(self, deep=True):
        """
        Retourne les paramètres pour cet estimateur.
        
        :param deep: Booléen, vrai par défaut. Si vrai, retournera les paramètres de `self` et des attributs sous-jacents.
        :return: Un dictionnaire des paramètres.
        """
        return {"model": self.model}
    
    def set_params(self, **parameters):
        """
        Définit les paramètres de cet estimateur.
        
        :param parameters: Un ou plusieurs paramètres sous forme de clés-valeurs à définir pour l'estimateur.
        :return: self, l'instance de l'estimateur avec les nouveaux paramètres.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
