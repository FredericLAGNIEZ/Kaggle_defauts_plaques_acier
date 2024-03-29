import argparse
import pandas as pd
import numpy as np
import sys
import os

# Récupérer le chemin d'accès du répertoire courant
current_dir = os.getcwd()
# Accéder au répertoire parent en utilisant os.pardir : on accède alors au repertoire scr
parent_current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

sys.path.append(parent_current_dir)

from src.data.load_data import load_csv_data
from src.data.clean_data import dropna, drop_duplicates
from src.data.split_data import split_data
from src.data.preprocess_data import standardize_data
from src.features.select_features import select_features_kbest,select_features_select_from_model,select_features_rfe
from src.models.model_random_forest import*
from src.models.model_logistic_regression import*

from src.models.model_gridsearch import*
from src.models.model_logistic_regression import*
from src.models.model_RandomizedSearchCV import*
from src.evaluation.model_evaluation import*
from src.models.multi_label import*
import mlflow
from src.evaluation.ml_flow import*
from src.utils.config import load_json

mlflow.set_experiment("Defauts_Plaques_Acier_Experiment")

parser = argparse.ArgumentParser(description="Pipeline d'exécution pour le projet DEFAUTS_PLAQUES_ACIER")
parser.add_argument('--load-data', action='store_true', help="Charge les données depuis le fichier CSV")
parser.add_argument('--clean-data', action='store_true', help="Nettoie les données chargés (doublons, valeurs manquantes)")
parser.add_argument('--split-data',action='store_true', help="Séparation du train en train et test")
parser.add_argument('--preprocess-data',action='store_true', help="Standardisation ou équivalent")
parser.add_argument('--feature-engineering',action='store_true', help="Création features")
parser.add_argument('--select-features', action='store_true', help="Activer la sélection de caractéristiques")
parser.add_argument('--method', type=str, default='select_from_model', choices=['select_kbest','select_from_model', 'rfe'], help="Méthode de sélection de caractéristiques à utiliser")
parser.add_argument('--model-random-forest', action='store_true', help="Entrainement random forest")
parser.add_argument('--model-regression-logistic', action='store_true', help="Entrainement Regression Logistique")
parser.add_argument('--model-gridsearch', action='store_true', help="Recherche de grille d'hyperpapramètres par Grid Search")
parser.add_argument('--model-randomsearch', action='store_true', help="Recherche de grille d'hyperpapramètres par Random Search")
parser.add_argument('--model-evaluation', action='store_true', help="Evaluation sur le test")

args = parser.parse_args()


def pipeline():
    with mlflow.start_run():

        if args.load_data: #chargement des données
            df = load_csv_data('train.csv')
             
        if args.clean_data: #nettoyage des données
            df = dropna(df)
            df = drop_duplicates(df)

        if args.split_data: #séparation des données
            """" Séparation du df chargé et nettoyé en train et test. Cela suppose connaitre les features et la target"""
            data = df.iloc[:,:-7]
            target = df.iloc[:,-7:]
            X_train,X_test,y_train,y_test=split_data(data,target, test_size=0.2)
        
        if args.preprocess_data: #outils de standardisation des données

            X_train,X_test,preprocessing_step= standardize_data(X_train,X_test)
            mlflow.log_param("preprocess", preprocessing_step)
        
        if args.select_features: #outils de sélection de réduction de dimension                
            
            if args.method =='select_kbest':
                X_train, X_test,selector = select_features_kbest(X_train, y_train, X_test, k=10) 
            elif args.method == 'select_from_model':
                X_train, X_test, selected_features,selector = select_features_select_from_model(X_train,X_test, y_train, model=None, threshold='mean')
                print(f"Caractéristiques sélectionnées : {selected_features}")
            elif args.method == 'rfe':
                X_train, X_test, selected_features,selector = select_features_rfe(X_train,X_test, y_train, n_features_to_select=10, model=None)
                print(f"Caractéristiques sélectionnées : {selected_features}")
            mlflow.log_param("selector", selector)            
        
        if args.model_random_forest:
                       
            model = RandomForestModel()
            model.train(X_train, y_train)
                       
            y_pred_proba = model.predict_proba(X_test)
            # Extraire la probabilité de la classe positive pour chaque cible et créer un array 2D
            prob_positives = np.array([proba[:, 1] for proba in y_pred_proba]).T  # Transposer pour avoir la forme correcte (n_samples, n_targets)
            y_pred_proba = pd.DataFrame(prob_positives, columns=y_test.columns)

            # Enregistrement des paramètres et des métriques dans MLflow
            params = {"model": model.name}
            params.update(model.model.get_params())
            metrics = {"multi_label_auc_scorer": multi_label_auc_scorer(y_test, y_pred_proba)}
            log_params_metrics(params, metrics)                   
                   
        if args.model_regression_logistic:
                       
            model = LogisticRegressionModel()
            model.train(X_train, y_train)
                       
            y_pred_proba = model.predict_proba(X_test)
            # Extraire la probabilité de la classe positive pour chaque cible et créer un array 2D
            prob_positives = np.array([proba[:, 1] for proba in y_pred_proba]).T  # Transposer pour avoir la forme correcte (n_samples, n_targets)
            y_pred_proba = pd.DataFrame(prob_positives, columns=y_test.columns)

            # Enregistrement des paramètres et des métriques dans MLflow
            params = {"model": model.name}
            params.update(model.model.get_params())
            metrics = {"multi_label_auc_scorer": multi_label_auc_scorer(y_test, y_pred_proba)}
            log_params_metrics(params, metrics)                   
                   
                   
        if args.model_gridsearch:
            #utilisation de la classe RandomForestClassifier de scikitlearn
            base_model = RandomForestClassifier() 
            wrapped_model = MultiLabelModelWrapper(base_model)

            param_grid = {
                                'n_estimators': [i for i in range(100,2000,100)],
                                'max_depth': [5, 10, None]
                            }
            
            grid_search=GridSearch(estimator=wrapped_model, param_grid=param_grid, cv=5, scoring=multi_label_auc)
            grid_search.fit(X_train,y_train)
        # Parcours des résultats de la recherche sur grille
            for i in range(len(grid_search.cv_results_['params'])):
                with mlflow.start_run(nested=True):  # Commence un sous-run pour chaque combinaison de paramètres
                    params = grid_search.cv_results_['params'][i]
                    score = grid_search.cv_results_['mean_test_score'][i]
            
                    # Logging des paramètres et du score moyen pour le run actuel
                    mlflow.log_params(params)
                    mlflow.log_metric("mean_test_score", score)
                    mlflow.end_run()  # Termine le sous-run

            y_pred_proba = grid_search.grid_search.best_estimator_.predict_proba(X_test)
            best_score = multi_label_auc_scorer(y_test,y_pred_proba)
            print(f"le meilleur modèle a pour score sur le test :{best_score}")

        if args.model_randomsearch:
            #utilisation de la classe RandomForestClassifier de scikitlearn
            base_model = RandomForestClassifier() 
            wrapped_model = MultiLabelModelWrapper(base_model)

            #import du fichier des hyperparamètres via un json
            config=load_json("random_forest_randomsearch.json")
            param_grid =config["RandomForestClassifier"]

            n_iter=100    
            random_search=RandomSearch(estimator=wrapped_model, param_grid=param_grid, n_iter=n_iter, cv=5, scoring=multi_label_auc)
            random_search.fit(X_train,y_train)
            
            # Parcours des résultats de la recherche sur grille
            for i in range(len(random_search.random_search.cv_results_['params'])):
                with mlflow.start_run(nested=True):  # Commence un sous-run pour chaque combinaison de paramètres
                    params = random_search.random_search.cv_results_['params'][i]
                    score = random_search.random_search.cv_results_['mean_test_score'][i]
            
                    # Logging des paramètres et du score moyen pour le run actuel
                    mlflow.log_params(params)
                    mlflow.log_metric("mean_test_score", score)
                    mlflow.end_run()  # Termine le sous-run

            y_pred_proba = random_search.random_search.best_estimator_.predict_proba(X_test)
            
            score = multi_label_auc_scorer(y_test,y_pred_proba)

            print(f"le meilleur modèle a pour score sur le test :{score}")
            
                     
        
        if args.model_evaluation:
            eval_test = ModelEvaluation(y_test, y_pred_proba)
            score = eval_test.average_auc()
            print(f"La moyenne des scores AUC est:{score}")
            mlflow.log_metric("average_auc", score)


if __name__ == "__main__":
    pipeline()
