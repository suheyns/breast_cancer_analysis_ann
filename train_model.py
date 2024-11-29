# train_model.py
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2
import joblib  # Import joblib for saving the model

# Load the breast cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Check for missing values
print("Checking for missing values...")
print(df.isnull().sum())

# Explore descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Normalize the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df.drop('target', axis=1))
scaled_df = pd.DataFrame(scaled_features, columns=data.feature_names)
scaled_df['target'] = df['target'].values

# Feature Selection
X = scaled_df.drop('target', axis=1)
y = scaled_df['target']
selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, selector.get_support(indices=True)], y, test_size=0.2, random_state=42)

# Grid Search CV for Model Tuning
mlp = MLPClassifier(max_iter=1000, random_state=42)
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (50, 50)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive'],
    'alpha': [0.0001, 0.001, 0.01],
    'batch_size': [10, 20, 40],
}

grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and score
print("\nBest Parameters:")
print(grid_search.best_params_)
print("\nBest Cross-Validation Score:")
print(grid_search.best_score_)

# Implementing the ANN Model
ann_model = MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', alpha=0.0001, learning_rate='constant', batch_size=20, max_iter=1000, random_state=42)
ann_model.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(ann_model, 'breast_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate the Model
y_pred = ann_model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))