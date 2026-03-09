#Implement ordinal encoding and one-hot encoding methods in Python from scratch.
import numpy as np
# 1. ORDINAL ENCODING (FROM SCRATCH)
# Assigns an integer rank to each category

class OrdinalEncoder:
    def __init__(self, order=None):
        """
        order: dict of {feature_index: [ordered_categories]}
               If None, order is assigned alphabetically
        """
        self.order = order
        self.mapping_ = {}       # category → integer
        self.inverse_mapping_ = {}  # integer → category

    def fit(self, X):
        X = np.array(X)
        num_cols = X.shape[1] if X.ndim > 1 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for col in range(num_cols):
            if self.order and col in self.order:
                categories = self.order[col]
            else:
                categories = sorted(set(X[:, col]))  # Alphabetical if no order given

            self.mapping_[col] = {cat: idx for idx, cat in enumerate(categories)}
            self.inverse_mapping_[col] = {idx: cat for idx, cat in enumerate(categories)}
        return self

    def transform(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_encoded = np.zeros(X.shape, dtype=int)

        for col in range(X.shape[1]):
            for row in range(X.shape[0]):
                val = X[row, col]
                if val not in self.mapping_[col]:
                    raise ValueError(f"Unknown category '{val}' in column {col}")
                X_encoded[row, col] = self.mapping_[col][val]
        return X_encoded

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_decoded = np.empty(X.shape, dtype=object)
        for col in range(X.shape[1]):
            for row in range(X.shape[0]):
                X_decoded[row, col] = self.inverse_mapping_[col][X[row, col]]
        return X_decoded

# 2. ONE-HOT ENCODING (FROM SCRATCH)
# Creates a binary column for each category

class OneHotEncoder:
    def __init__(self):
        self.categories_ = {}      # col → list of unique categories
        self.feature_names_ = []   # output column names

    def fit(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.feature_names_ = []
        for col in range(X.shape[1]):
            unique_cats = sorted(set(X[:, col]))
            self.categories_[col] = unique_cats
            for cat in unique_cats:
                self.feature_names_.append(f"col{col}_{cat}")
        return self

    def transform(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        encoded_cols = []
        for col in range(X.shape[1]):
            for cat in self.categories_[col]:
                binary_col = (X[:, col] == cat).astype(int)
                encoded_cols.append(binary_col)

        return np.column_stack(encoded_cols)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_names(self):
        return self.feature_names_

#TESTS
print("ORDINAL ENCODING - FROM SCRATCH")

# Sample data with natural order
data_ordinal = [
    ["Low",    "Small"],
    ["High",   "Large"],
    ["Medium", "Medium"],
    ["Low",    "Large"],
    ["High",   "Small"],
]

# Define explicit order for each column
order = {
    0: ["Low", "Medium", "High"],    # col 0: Low < Medium < High
    1: ["Small", "Medium", "Large"]  # col 1: Small < Medium < Large
}

oe = OrdinalEncoder(order=order)
encoded = oe.fit_transform(data_ordinal)

print("\nOriginal Data:")
for row in data_ordinal:
    print(f"  {row}")

print("\nOrdinal Encoded:")
print(f"  {'Category':<30} {'Encoded'}")
print(f"  {'-'*40}")
for orig, enc in zip(data_ordinal, encoded):
    print(f"  {str(orig):<30} {list(enc)}")

print("\nMapping Used:")
print(f"  Col 0 (Rank)  : {oe.mapping_[0]}")
print(f"  Col 1 (Size)  : {oe.mapping_[1]}")

print("\nInverse Transform (back to original):")
decoded = oe.inverse_transform(encoded)
for row in decoded:
    print(f"  {list(row)}")

print("ONE-HOT ENCODING - FROM SCRATCH")

# Sample data
data_ohe = [
    ["Red",   "Cat"],
    ["Blue",  "Dog"],
    ["Green", "Cat"],
    ["Red",   "Bird"],
    ["Blue",  "Dog"],
]

ohe = OneHotEncoder()
ohe_encoded = ohe.fit_transform(data_ohe)

print("\nOriginal Data:")
for row in data_ohe:
    print(f"  {row}")

print(f"\nFeature Names: {ohe.get_feature_names()}")
print("\nOne-Hot Encoded Matrix:")
print(f"  {' '.join(f'{n:>10}' for n in ohe.get_feature_names())}")
print(f"  {'-' * 75}")
for orig, enc in zip(data_ohe, ohe_encoded):
    print(f"  {' '.join(f'{v:>10}' for v in enc)}   ← {orig}")