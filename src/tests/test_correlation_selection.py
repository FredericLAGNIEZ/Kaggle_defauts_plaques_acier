import unittest
import pandas as pd

import sys
sys.path.append('/run/media/fredericlagniez/Echanges_Linux_Windows/GitHUb/defauts_acier/defauts_plaques_acier/src/features')
from select_features import correlation_selection,load_raw_data

#Chargement du DataFrame
df= load_raw_data("/train.csv")
#Séparation des variables et des cibles
features = df.iloc[:,:-7]
targets = df.iloc[:,-7:]

class TestCorrelationSelection(unittest.TestCase):

    def test_correlation_selection(self):
        # Créer un dataframe de test
        data = {'feature_1': [1, 2, 3, 4, 5],
                'feature_2': [2, 3, 4, 5, 6],
                'feature_3': [0.1, 0.2, 0.3, 0.4, 0.5],
                'target': [1, 0, 1, 0, 0]}
        df = pd.DataFrame(data)

        # Appeler la fonction de sélection de caractéristiques
        selected_features = correlation_selection(df,features=features,targets=targets,target="Pastry")

        # Vérifier que les caractéristiques sélectionnées sont correctes
        expected_features = ['feature_1', 'feature_2']
        self.assertEqual(set(selected_features), set(expected_features))

if __name__ == '__main__':
    unittest.main()
