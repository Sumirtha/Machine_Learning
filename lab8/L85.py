#Use breast_cancer.csv (https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv)
#Use scikit learn methods, OrdinalEncoder, OneHotEncoder(sparse=False), LabelEncoder to implement complete Logistic Regression Model.

# load and summarize the dataset
from pandas import read_csv
# define the location of the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
# load the dataset
dataset = read_csv(url, header=None)
# retrieve the array of data
data = dataset.values
# separate into input and output columns
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
# summarize
print('Input', X.shape)
print('Output', y.shape)