from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif,mutual_info_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import collections
import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tabulate
import math
import collections
from sklearn.utils import check_X_y
import mlflow

from sklearn.feature_selection import (
    SelectKBest, 
    chi2, 
    f_classif, 
    f_regression,
    r_regression,
    mutual_info_classif,
    mutual_info_regression
)
"""
dossier_github_frederic = "/media/frederic/Echanges_Linux_Windows/GitHUb/defauts_acier/defauts_plaques_acier"

from src.data.load_data import load_csv_data

#Chargement du DataFrame
df= load_csv_data(data_filename="train.csv",data_folder_path=dossier_github_frederic)
#Séparation des variables et des cibles
features = df.iloc[:,:-7]
targets = df.iloc[:,-7:]
"""
# Méthodologie de sélection des features

def correlation_selection(df,features,targets,target,correlation_threshold=0.1):
    """
    Sélectionne les caractéristiques numériques parmi les feature du dataframe df
    qui ont une correlation avec la cible (target) supérieure au seuil de selection
    (threshold).
    """
    #Concaténation features et target
    concat = pd.concat([features,targets[target]],axis=1)
    
    #Matrice de corrélation
    correlations= concat.corr()[target]

    selected_features = correlations[abs(correlations) > correlation_threshold]

    # Retirer la cible
    remove_target = selected_features.index[selected_features.index != target]

    return selected_features[remove_target].index



def select_features_kbest(df, features, target, method, k=20):
    """
    Sélectionne à l'aide de la méthode KBest les k meilleures caractéristiques selon
    différentes méthodes statistiques, avec k=20 par défaut
    Retourne un dataframe contenant les k variables les plus pertinentes
    """
    # Sélectionner les k meilleures caractéristiques
    selector = SelectKBest(method, k=k)
    selected_features = selector.fit_transform(features, df[target])

    # Transformation en DataFrame
    selected_features = pd.DataFrame(selected_features)

    # Renommer les colonnes en utilisant les noms des colonnes sélectionnées
    selected_features.columns = selector.get_feature_names_out()

    # Sélectionner les colonnes sélectionnées
    selected_columns = selected_features.columns

    return df[selected_columns].columns

def select_features_select_from_model(X_train,X_test, y_train, model=None, threshold='mean'):
    """
    Sélectionne les caractéristiques en utilisant SelectFromModel.

    :param X_train: Features d'entrainement.
    :param y_train: Target d'entraînement.
    :param model: Modèle à utiliser pour l'importance des caractéristiques. Si None, RandomForestClassifier est utilisé.
    :param threshold: Seuil pour la sélection des caractéristiques ('mean', 'median', ou une valeur flottante).
    :return: X_train avec les caractéristiques sélectionnées, X_test avec les caractéristiques sélectionnées et  liste des noms des caractéristiques sélectionnées.
    """
    if model is None:
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()]
    
    return X_train_selected,X_test_selected, selected_features,"select_from_model"


def select_features_rfe(features,target,n_features_to_select=20, model=None):
    """
    Sélectionne les k meilleures caractéristiques en utilisant RFE.

    :param features: Features d'entrainement.
    :param target: Target d'entraînement.
    :param n_features_to_select: Nombre de caractéristiques à sélectionner.
    :param model: Modèle à utiliser pour l'évaluation des caractéristiques. Si None, RandomForestClassifier est utilisé.
    :return les colonnes conservées à la suite de la selection 
    """
    target=targets[target]
    
   # Vérification des types et des valeurs des arguments en entrée
    if not isinstance(features, (np.ndarray, pd.DataFrame)):
        raise ValueError("features doit être un tableau NumPy ou un DataFrame Pandas")
    if not isinstance(target, (np.ndarray, pd.Series)):
        raise ValueError("target doit être un tableau NumPy ou une série Pandas")
    if not isinstance(n_features_to_select, int) or n_features_to_select <= 0:
        raise ValueError("n_features_to_select doit être un entier positif")

    # Vérification des dimensions des données d'entrée
    features, target = check_X_y(features, target)  
   
    if model is None:
        model = RandomForestClassifier()
    
    #instanciation    
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    
    features_new=rfe.fit_transform(features,target)
    
    features_new=pd.DataFrame(features_new)
    
    features_new.columns = rfe.get_feature_names_out()  
    
    return features_new.columns

""" old :
def select_features_rfe(features, target, n_features_to_select=20, model=None):
    "
    Sélectionne les k meilleures caractéristiques en utilisant RFE.

    :param X_train: Features d'entrainement.
    :param y_train: Target d'entraînement.
    :param n_features_to_select: Nombre de caractéristiques à sélectionner.
    :param model: Modèle à utiliser pour l'évaluation des caractéristiques. Si None, RandomForestClassifier est utilisé.
    :return: X_train avec les caractéristiques sélectionnées, X_test avec les caractéristiques sélectionnées et  liste des noms des caractéristiques sélectionnées.
        if model is None:
        model = RandomForestClassifier()
        rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    
    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)
    selected_features = X_train.columns[rfe.support_]
    return X_train_selected,X_test_selected, selected_features,"rfe"
"""

#Comptage des variables non utilisés

def count_useless_features(df,features,targets,method,correlation_threshold=0.1):
       
    # Créer un defaultdict avec une valeur initiale de 0
    dictionnaire = collections.defaultdict(int)
    """
    Pour chaque target :
        - sélectionne les caractéristiques numériques parmi les feature du dataframe df
    qui ont une correlation avec la cible (target) supérieure au seuil de selection
    (threshold).
        - Un dictionnaire est utilisé pour compter le nombre d'occurence de chaque variable
    non utilisée.
    Le dictionnaire final est trié par ordre décroissant
    """
    
    # Ajouter une entrée pour chaque clé de la liste avec une valeur de 0
    for cle in features:
        dictionnaire[cle] = 0

    for target in targets.columns:
        if method=="correlation":            

            selected_features = correlation_selection(df=df,
                                                    features=features,
                                                    targets=targets,target=target,
                                                    correlation_threshold=correlation_threshold)
        elif method == "f_classif": 

            selected_features=select_features_kbest(df=df,features=features,target=target,
                                                    method=f_classif,
                                                    k=20)
        elif method == "mutual_info_classif":

            selected_features=select_features_kbest(df=df,features=features,target=target,
                                                    method=mutual_info_classif,
                                                    k=20)
        
        elif method == "rfe":
    
            selected_features=select_features_rfe(features=features,target=target,n_features_to_select=20)
            
        else:
            print("Vous n'avez pas choisi une méthode appropriée.")

        #Variables non selectionnées                
        non_used_features = [x for x in features if x not in selected_features]
            
        # Augmenter le compteur de 1 pour chaque variable non utilisée
        for cle in non_used_features:
                dictionnaire[cle] += 1            
     
    # Trier le dictionnaire par les valeurs
    dictionnaire_trie = sorted(dictionnaire.items(), key=lambda x: x[1],reverse=True)

    # Convertir la liste de tuples triés en dictionnaire
    dictionnaire_useless_features = dict(dictionnaire_trie)    

    return dictionnaire_useless_features


"""
def select_features_kbest(X_train, y_train, X_test, k=20):
    
    Sélectionne les k meilleures caractéristiques, avec k=20 par défaut
    
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    print(f"les {k} features les plus pertinentes ont été sélectionnées.")

    return X_train_selected, X_test_selected,"selectKbest"

"""



#Programme principal
"""
Conduit à redéfinir les features qui seront utilisées, en fonction du nombre de variables qu'il est choisi de supprimer


if __name__ == '__main__':  
  
         
    while True:
        try:
            nombre = int(input("Entrez un nombre entier compris entre 1 et 27 pour selectionner autant de variables: "))
            if 1 <= nombre <= 27:
                print(f"Vous avez choisi de selectionner {nombre} caratéristiques.")
                break
            else:
                print("Votre nombre n'est pas compris entre 1 et 27. Veuillez réessayer.")
        except ValueError:
            print("Vous n'avez pas entré un nombre entier. Veuillez réessayer.")   

    while True:
        try:
            method = str(input("Quelle méthode voulez vous utiliser en premier pour commencer à compter les variables non utilisées : correlation,f_classif ou mutual_info_classif?"))
            if method in ["correlation","f_classif","mutual_info_classif"]:
                print(f"Vous avez choisi la méthode {method}. Patience, je calcule...",end="\n\n")
                break
            else:
                print("Vous n'avez pas choisi une méthode disponible. Veuillez réessayer.")
        except ValueError:
            print("Vous n'avez pas entré une chaîne de caractères. Veuillez réessayer.")

    dictionnaire_useless_features = count_useless_features(df=df,features=features,targets=targets, method=method,correlation_threshold=0.1)
    print(f"Voici les variables les moins utilisées par la méthode {method}] :", dictionnaire_useless_features, end="\n\n")
    
    
    print("Je réalise les calcules pour l'ensemble des 4 méthodes de sélection, patientez... ",end="\n")

    # création des dictionnaires correspondant à chaque mêthode
    dictionnaire_trie = dico_f_classif_trie=count_useless_features(df=df,features=features,targets=targets, method="correlation",correlation_threshold=0.1)
    dico_f_classif_trie=count_useless_features(df=df,features=features,targets=targets, method="f_classif",correlation_threshold=0.1)
    dico_mutual_info_trie = count_useless_features(df=df,features=features,targets=targets, method="mutual_info_classif",correlation_threshold=0.1) 
    dico_rfe_trie=count_useless_features(df=df,features=features,targets=targets,method="rfe",correlation_threshold=0.1) 
    
    print("je vais additionner les 4 dictionnaires... ",end="\n")
    
    result = {}
    # Boucle pour parcourir les clés des quatre dictionnaires
    for key in dictionnaire_trie:
    # Ajout des valeurs correspondantes dans le résultat
        result[key] = dictionnaire_trie[key] + dico_f_classif_trie[key] + dico_mutual_info_trie[key] + dico_rfe_trie[key]

    # Tri du résultat en fonction des valeurs
    result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        
    print("\n\n Voici les variable les moins utilisées pour l'ensemble des trois méthodes sont : ", result)
    
    nombre = int(input("Combien de variables voulez-vous éliminer par les vaiables les moins selectionnées ? Entrez un nombre entier compris entre 1 et 26: "))
    print(f"Vous avez choisi d'éliminer {nombre} caratéristiques.")

    # Récupération des nombre premières clés du dictionnaire
    useless_features = list(result.keys())[:nombre]
    
    print(f"Voici les {nombre} variables les moins utilisées : ",useless_features)   
        
    # Suppression des variables non utilisées dans le dataframe
    selected_features = features.drop(useless_features, axis=1)

    # enregistrement des caractéristiques sélectionnées dans un fichier CSV
    selected_features.to_csv( "/media/frederic/Echanges_Linux_Windows/GitHUb/defauts_acier/defauts_plaques_acier/data_to_use" + '/selected_features.csv', index=False)"""