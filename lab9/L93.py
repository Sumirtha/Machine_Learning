#Implement a classification decision tree algorithm using scikit-learn for the sonar dataset.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Sonar dataset from URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/sonar/sonar.all-data"
data = pd.read_csv(url, header=None)

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert labels (R=Rock, M=Mine) to numbers
y = y.map({'R':0, 'M':1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)