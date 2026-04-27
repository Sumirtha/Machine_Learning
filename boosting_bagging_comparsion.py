#A1. Apply boosting..........

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
#Loading data
X,y = make_hastie_10_2(random_state=69)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2000, random_state=69)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)
#Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def a():
    Grad_Boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=69) #100 Decision stumps taken as weak learners
    Grad_Boost.fit(X_train, y_train)
    y_pred = Grad_Boost.predict(X_test)
    print("\nAccuracy score for Gradient Boosting: ", accuracy_score(y_test, y_pred))
    print("\nF1 score for Gradient Boosting: ", f1_score(y_test, y_pred, average='weighted'))

def b():
    bagging = BaggingClassifier(DecisionTreeClassifier())
    bagging.fit(X_train, y_train)
    y_pred = bagging.predict(X_test)
    print("\nAccuracy score for Decision tree classifier: " ,accuracy_score(y_test, y_pred))
    print("\nF1 score for Decision tree classifier: ",f1_score(y_test, y_pred, average='weighted'))

def c():

    svc = SVC(random_state=69)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print("\nAccuracy score for SVC: ",accuracy_score(y_test, y_pred))
    print("\nF1 score for SVC : ", f1_score(y_test, y_pred, average='weighted'))
    print("\nThe results are almost accurate compared to SVC, as difference is .08")

def d():
    print("\nSVC yields better performance compared to boosting and bagging as the f1 score of SVC is 0.961 ")

def e():
    print("\nBoosting has Accuracy score of 0.8812 and f1 score of 0.8809718284597801"
          "\nBagging has Accuracy score of 0.804 and F1 score of 0.804"
          "\nWhen compared between these two, Boosting is a better model in performance.")


def main():
    a()
    b()
    c()
    d()
    e()

if __name__ == '__main__':
    main()
