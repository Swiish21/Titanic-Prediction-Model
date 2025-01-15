from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib

def train_model(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Prepare features and labels
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Validate the model
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    
    # Save the trained model
    joblib.dump(model, 'titanic_model.pkl')

if __name__ == "__main__":
    train_model('../data/train.csv')