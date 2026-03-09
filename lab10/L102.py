#Implement information gain measures. The function should accept data points for parents, data points for both children and return an information gain value.

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


def information_gain(parent, left_child, right_child):
    total = len(parent)

    parent_entropy = entropy(parent)
    left_weight = len(left_child) / total
    right_weight = len(right_child) / total

    ig = parent_entropy - (left_weight * entropy(left_child) + right_weight * entropy(right_child))
    return ig


def main():
    parent = ['Yes', 'Yes', 'No', 'No', 'Yes', 'No']
    left_child = ['Yes', 'Yes', 'No']
    right_child = ['No', 'Yes', 'No']

    print("Information Gain:", information_gain(parent, left_child, right_child))


if __name__ == "__main__":
    main()