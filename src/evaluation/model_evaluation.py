from sklearn.metrics import roc_auc_score
from sklearn.metrics import  make_scorer
import numpy as np


class ModelEvaluation:
    def __init__(self, true_labels, predictions):
        """
        Initialisation de la classe d'évaluation.
        :param true_labels: les étiquettes réelles.
        :param predictions: les probabilités prédites par le modèle.
        """
        self.true_labels = true_labels
        self.predictions = predictions

    def calculate_auc(self):
        """
        Calculer l'AUC pour chaque catégorie de défaut.
        :return: dictionnaire des AUC pour chaque défaut.
        """
        auc_scores = {}
        for i,column in enumerate(self.true_labels.columns):
            auc = roc_auc_score(self.true_labels[column], self.predictions[column])
            auc_scores[column] = auc
        return auc_scores

    def average_auc(self):
        """
        Calculer la moyenne des scores AUC sur toutes les catégories de défaut.
        :return: moyenne des AUC.
        """
        auc_scores = self.calculate_auc()
        average_auc = sum(auc_scores.values()) / len(auc_scores)
        return average_auc



def multi_label_auc_scorer(y_true, y_pred):
    """
    Calcule le score AUC moyen pour un problème multi-label.
    
    :param y_true: Les vraies étiquettes (un tableau 2D avec des indicateurs binaires).
    :param y_pred: Les scores ou probabilités prédites pour chaque label (également un tableau 2D).
    :return: Le score AUC moyen.
    """
    # Assurez-vous que y_true et y_pred sont des numpy arrays
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
        
    # Calculer le score AUC pour chaque label et prendre la moyenne
    auc_scores = [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return np.mean(auc_scores)


from sklearn.metrics import make_scorer

# Créer un scorer scikit-learn à partir de la fonction de scoring personnalisée
multi_label_auc = make_scorer(multi_label_auc_scorer, needs_proba=True)
