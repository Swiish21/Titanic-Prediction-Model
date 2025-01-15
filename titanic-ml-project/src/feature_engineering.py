def create_features(df):
    # Create a new feature 'FamilySize' by combining 'SibSp' (siblings/spouses) and 'Parch' (parents/children) plus 1 (the passenger themselves)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create a new feature 'IsAlone' which indicates if the passenger is alone (1) or not (0)
    df['IsAlone'] = 1  # Initialize to 1 (True)
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0  # Set to 0 (False) if FamilySize > 1
    
    # Encode categorical variables 'Sex' and 'Embarked' into dummy/indicator variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    return df

def scale_features(df):
    from sklearn.preprocessing import StandardScaler
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # List of numerical features to be scaled
    numerical_features = ['Age', 'Fare', 'FamilySize']
    
    # Fit the scaler on the numerical features and transform them
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df