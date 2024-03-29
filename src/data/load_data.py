import pandas as pd
from pathlib import Path
import os

def load_csv_data(data_filename):
    """
    Charge les données depuis un fichier CSV situé dans le dossier 'data' ou un de ses sous-dossiers.
    
    :param data_filename: str, le nom du fichier CSV à charger avec son extension .csv.
    :param data_folder: str, le chemin du dossier parent du projet.
    :return: pandas.DataFrame, les données chargées, ou None si le fichier n'est pas trouvé.
    """
    # Récupérer le chemin d'accès du répertoire courant
    current_dir = os.getcwd()
    
    # Accéder au répertoire parent en utilisant os.pardir : on accède alors au repertoire scr
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    
    # Accéder de nouveau au répertoire parent en utilisant os.pardir : on accède alors au repertoire du projet lui-même
    grand_parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))

    # Associer à une variable le chemin d'accès absolu du repertoire du projet
    data_project_path = Path(grand_parent_dir)

    # Rechercher le fichier dans le dossier et ses sous-dossiers de ce projet
    file_list = list(data_project_path.rglob(data_filename))
    
    # Vérifier si au moins un fichier correspondant au nom a été trouvé
    if file_list:
        # Charger les données depuis le premier fichier trouvé
        df = pd.read_csv(file_list[0], index_col=0)
        print(f"Data loaded successfully from {file_list[0]}")
        return df
    else:
        print(f"Le fichier {data_filename} n'a pas été trouvé dans le repertorie parent {data_project_path} ou ses sous-dossiers.")
        return None