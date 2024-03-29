from sklearn.preprocessing import StandardScaler
import pandas as pd

def standardize_data(X_train,X_test):
    """
    Standardise les features numériques d'un DataFrame.
    :param df: pandas.DataFrame
    :return: pandas.DataFrame standardisé
    """
    
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print("Standardisation effectuée")
    return X_train,X_test,"standardisation"
