#Write a program to partition a dataset (simulated data for regression)  into two parts, based on a feature (BP) and for a threshold, t = 80.
#Generate additional two partitioned datasets based on different threshold values of t = [78, 82].
import pandas as pd


def partition_dataset(data, threshold):
    lower = data[data['BP'] <= threshold]
    higher = data[data['BP'] > threshold]

    print("\nThreshold:", threshold)
    print("Records with BP <=", threshold)
    print(lower)

    print("\nRecords with BP >", threshold)
    print(higher)


def main():
    # Load dataset
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

    # Partition using different thresholds
    partition_dataset(data, 80)
    partition_dataset(data, 78)
    partition_dataset(data, 82)


if __name__ == "__main__":
    main()