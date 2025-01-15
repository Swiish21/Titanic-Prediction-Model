import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    # Load the dataset from the specified file path
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(data):
    # Handle missing values by forward filling them
    data.fillna(method='ffill', inplace=True)
    return data

def preprocess_data(data):
    # Preprocess the data by handling missing values
    data = handle_missing_values(data)
    # Additional preprocessing steps can be added here
    return data

def split_data(data, target_column):
    # Split the data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test