import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Heart Disease Classification

def heart_classifier():

    # Load dataset
    df = pd.read_csv("/home/ibab/PyCharmMiscProject/ML/ALL+CSV+FILES+-+2nd+Edition+-+corrected/ALL CSV FILES - 2nd Edition/Heart.csv")

    df = df.drop("Unnamed: 0", axis=1)

    # Convert target variable
    df["AHD"] = df["AHD"].map({"No":0, "Yes":1})

    # Encode categorical features
    df = pd.get_dummies(df, drop_first=True)
    df = df.dropna()
    X = df.drop("AHD", axis=1)
    y = df["AHD"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=69)

    # Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Probabilities
    probs = model.predict_proba(X_test)[:,1]

    thresholds = [0.3, 0.5, 0.7]

    for t in thresholds:

        pred = (probs >= t).astype(int)

        TP = np.sum((pred==1) & (y_test==1))
        TN = np.sum((pred==0) & (y_test==0))
        FP = np.sum((pred==1) & (y_test==0))
        FN = np.sum((pred==0) & (y_test==1))

        accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        sensitivity = TP/(TP+FN) if (TP+FN) > 0 else 0
        specificity = TN/(TN+FP) if (TN+FP) > 0 else 0
        f1 = 2*(precision*sensitivity)/(precision+sensitivity) if (precision+sensitivity) > 0 else 0

        print("\nThreshold:",t)
        print("Accuracy:",accuracy)
        print("Precision:",precision)
        print("Sensitivity:",sensitivity)
        print("Specificity:",specificity)
        print("F1 Score:",f1)

    # ROC Curve
    fpr, tpr, thr = roc_curve(y_test, probs)

    roc_auc = auc(fpr,tpr)

    plt.plot(fpr,tpr,label="ROC curve (AUC = %0.2f)"%roc_auc)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    print("\nAUC:",roc_auc)

def main():
    heart_classifier()

if __name__ == "__main__":
    main()