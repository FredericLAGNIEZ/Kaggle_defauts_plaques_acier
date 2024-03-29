# Steel Plate Defect Prediction
 Competition Kaggle - Steel Plate Defect Prediction - Mars 2024 - Playground Series - Season 4, Episode 3

# commande pour ajouter ce chemin au PATH pour réussir l'import des modules

$env:PYTHONPATH += ";C:\Users\franc\DATA\DATA_Projet\Kaggle\defauts_plaques_acier"
& C:/Users/franc/anaconda3/python.exe c:/Users/franc/DATA/DATA_Projet/Kaggle/defauts_plaques_acier/scripts/main.py


# exemple de ligne de commande dans /script pour exécuter une pipeline du main:

python main.py --load-data --clean-data --split-data --preprocess-data --model-random-forest --model-evaluation

# acces à l'interface de ML Flow

mlflow ui

