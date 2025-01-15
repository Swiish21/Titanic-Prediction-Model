import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data, preprocess_data, split_data
from feature_engineering import create_features, scale_features
from model_evaluation import evaluate_model

# Load and preprocess the training data
train_data = load_data('../data/train.csv')
train_data = preprocess_data(train_data)
train_data = create_features(train_data)
train_data, _ = scale_features(train_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(train_data, 'Survived')

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
evaluation_results = evaluate_model(y_test, y_pred)
print(evaluation_results)

# Load and preprocess the test data
test_data = load_data('../data/test.csv')
test_data = preprocess_data(test_data)
test_data = create_features(test_data)
test_data, _ = scale_features(test_data)

# Make predictions on the test data
test_predictions = model.predict(test_data)

# Save the predictions to a CSV file
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions})
output.to_csv('../data/predictions.csv', index=False)