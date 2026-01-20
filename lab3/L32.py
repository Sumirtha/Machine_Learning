import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    # Loading data
#    df_data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    df_data = pd.read_csv("/home/ibab/PyCharmMiscProject/ML/lab3/simulated_data_multiple_linear_regression_for_ML.csv")

    # Printing the head
    print(df_data.head())
    # FIXED: Separate features and target properly
    # Assuming 'disease_score_fluct' is your target variable
    X = df_data.drop('disease_score_fluct', axis=1)  # All columns except target
    y = df_data['disease_score_fluct']  # Target column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    # Standardizing the data
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)
    # Initializing model
    model = LinearRegression()
    # Training a model
    model.fit(X_train_scale, y_train)
    y_pred = model.predict(X_test_scale)
    r2 = r2_score(y_test, y_pred)
    print("r2 value:", r2)
    # print("Model_score:", model.score(X_test_scale, y_test))
    print("Done")

if __name__ == '__main__':
    main()