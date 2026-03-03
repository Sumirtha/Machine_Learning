#Perform 10-fold cross validation for SONAR dataset in scikit-learn using logistic regression.
#SONAR dataset is a binary classification problem with target variables as Metal or Rock. i.e. signals are from metal or rock.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. LOAD SONAR DATASET
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
df = pd.read_csv(url, header=None)

print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution:\n{df[60].value_counts()}")

# 2. PREPARE FEATURES AND LABELS
X = df.iloc[:, :60].values      # 60 sonar frequency features
y = df.iloc[:, 60].values       # Target: 'M' (Metal) or 'R' (Rock)

# Encode labels: M → 1, R → 0
le = LabelEncoder()
y = le.fit_transform(y)
print(f"\nLabel Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3. SCALE FEATURES
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. 10-FOLD CROSS VALIDATION
kf = KFold(n_splits=10, shuffle=True, random_state=69)
model = LogisticRegression(max_iter=1000, random_state=69)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# 5. RESULTS
print("     10-Fold Cross Validation       ")

for i, score in enumerate(scores, 1):
    print(f"  Fold {i:02d}: Accuracy = {score:.7f}")

print(f"  Mean Accuracy : {np.mean(scores):.7f}")
print(f"  Std Deviation : {np.std(scores):.7f}")
print(f"  Min Accuracy  : {np.min(scores):.7f}")
print(f"  Max Accuracy  : {np.max(scores):.7f}")