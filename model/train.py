from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv

def load_data_X_y():
    X = []
    y = []
    with open('model/sal.csv', 'r') as file:
        my_reader = csv.reader(file, delimiter=',')
        for row in my_reader:
            row =[float(r) for r in row]
            X.append(row[:-1])
            y.append(row[-1])
    print (X, y)
    return X, y

def get_train_test_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    return X_train, X_test, y_train, y_test


def train_and_return_model(X_train, y_train):
    
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    return lm
    
def score_model(lm, X_test, y_test):
    preds = lm.predict(X_test)
    print ("Preds ", preds)
    return mean_squared_error(preds, y_test)
