#Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch.
import numpy as np
def normalize_min_max(data):
    """
    Min-Max Normalization: Scale values to range [0, 1]

    Formula: X_normalized = (X - X_min) / (X_max - X_min)

    Parameters:
    -----------
    data : list or numpy array
        Input data to normalize

    Returns:
    --------
    normalized : numpy array
        Normalized data in range [0, 1]
    min_val : float or array
        Minimum value(s) used for normalization
    max_val : float or array
        Maximum value(s) used for normalization
    """
    # Convert to numpy array if needed
    data = np.array(data, dtype=float)

    # Find min and max
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    # Calculate range (avoid division by zero)
    range_val = max_val - min_val

    # Handle case where all values are the same
    if np.isscalar(range_val):
        if range_val == 0:
            range_val = 1
    else:
        range_val = np.where(range_val == 0, 1, range_val)

    # Apply normalization
    normalized = (data - min_val) / range_val

    return normalized, min_val, max_val

def denormalize(normalized_data, min_val, max_val):
    """
    Reverse the normalization to get original values

    Formula: X_original = X_normalized * (X_max - X_min) + X_min

    Parameters:
    -----------
    normalized_data : numpy array
        Normalized data in range [0, 1]
    min_val : float or array
        Original minimum value(s)
    max_val : float or array
        Original maximum value(s)

    Returns:
    --------
    original : numpy array
        Denormalized data (original scale)
    """
    original = normalized_data * (max_val - min_val) + min_val
    return original

data2 = np.array([
    [100, 5, 1000],
    [200, 10, 2000],
    [150, 7, 1500],
    [250, 12, 2500]
])

print("Original data:")
print(data2)

normalized2, min2, max2 = normalize_min_max(data2)
print("\nNormalized data (each column scaled to [0,1]):")
print(normalized2)
print(f"\nMin values: {min2}")
print(f"Max values: {max2}")