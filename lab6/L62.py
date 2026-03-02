def normalize(data):
    """
    Min-Max Normalization: scales all values between 0 and 1
    Formula: x_norm = (x - x_min) / (x_max - x_min)
    """
    min_val = min(data)
    max_val = max(data)

    if max_val == min_val:
        return [0.0 for _ in data]

    return [(x - min_val) / (max_val - min_val) for x in data]


def normalize_2d(data):
    """
    Normalize each column independently for 2D data (list of lists)
    """
    # Transpose to work column-wise
    num_cols = len(data[0])
    normalized = [[0.0] * num_cols for _ in range(len(data))]

    for col in range(num_cols):
        column = [data[row][col] for row in range(len(data))]
        norm_col = normalize(column)
        for row in range(len(data)):
            normalized[row][col] = norm_col[row]

    return normalized

# TEST: 2D Data (rows = samples, cols = features)
data_2d = [
    [1, 200, 0.5],
    [2, 400, 1.5],
    [3, 600, 2.5],
    [4, 800, 3.5]
]

print("\nOriginal 2D:")
for row in data_2d:
    print(row)

print("\nNormalized 2D:")
for row in normalize_2d(data_2d):
    print([round(x, 4) for x in row])