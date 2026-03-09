#Implement entropy measure using Python. The function should accept a set of data points and their class labels and return the entropy value.

import math
from collections import Counter


def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    ent = 0

    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)

    return ent


def main():
    data = ['Yes', 'Yes', 'No', 'No', 'Yes', 'No']
    print("Entropy:", entropy(data))


if __name__ == "__main__":
    main()