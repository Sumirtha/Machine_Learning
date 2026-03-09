#Build a classification model for wisconsin dataset using Ridge and Lasso classifier using scikit-learn

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():

    # Load Wisconsin dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Ridge Classifier
    ridge_model = RidgeClassifier()
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)

    # Lasso Classifier (using logistic regression with L1 penalty)
    lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)

    # Accuracy
    print("Ridge Accuracy:", accuracy_score(y_test, ridge_pred))
    print("Lasso Accuracy:", accuracy_score(y_test, lasso_pred))

if __name__ == "__main__":
    main()