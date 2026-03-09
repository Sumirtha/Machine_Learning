#Implement a regression decision tree algorithm using scikit-learn for the simulated dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load simulated dataset
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

# Features and target
X = data[['age','BMI','BP','blood_sugar','Gender']]
y = data['disease_score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create regression decision tree model
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Predicted values:", y_pred)