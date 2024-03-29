def dropna(df):
    """Supprimer les lignes avec des valeurs manquantes."""
    df = df.dropna()
    print("Valeurs manquantes supprimées")
    return df

def drop_duplicates(df):
    """Supprimer les lignes en double."""
    df = df.drop_duplicates()
    print("Doublons supprimés")
    return df