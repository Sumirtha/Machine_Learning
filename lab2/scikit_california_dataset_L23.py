from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    #loading data
    [X, y] = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
#standardizing the data
    scaler = StandardScaler()
    scaler=scaler.fit(X_train)
    X_train_scale=scaler.transform(X_train)
    X_test_scale=scaler.transform(X_test)
#initializing model
    model=LinearRegression()
#training a model
    model.fit(X_train_scale, y_train)
    y_pred=model.predict(X_test_scale)
    r2=r2_score(y_test, y_pred)
    print("r2 value:", r2)
#   print("Model_score:",model.score(X_test, y_test))
    print("Done")
if __name__=='__main__':
    main()