from sklearn.model_selection import train_test_split


def split_data(data,target,test_size=0.2):
    """Divise les données en ensembles d'entraînement et de test."""
    X_train,X_test,y_train,y_test = train_test_split(data,target, test_size=test_size, random_state=42)
    print("Séparation des données effectuée:")
    print(f"Shape X_train : {X_train.shape}")
    print(f"Shape X_test : {X_test.shape}")
    print(f"Shape y_train : {y_train.shape}")
    print(f"Shape y_test : {y_test.shape}")
    return X_train,X_test,y_train,y_test