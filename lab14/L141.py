#Implement Adaboost classifier using scikit-learn. Use the Iris dataset.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
def adaboost_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69)

    model = AdaBoostClassifier(n_estimators=50, random_state=69)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def main():
    adaboost_iris()

if __name__ == "__main__":
    main()