import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(data):
    data.fillna(method='ffill', inplace=True)
    return data

def preprocess_data(data):
    data = handle_missing_values(data)
    # Additional preprocessing steps can be added here
    return data

def split_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test