import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("sonar_data.csv")

#print(dataset.head())

#print(dataset.describe())

X = dataset.iloc[:, :-1]   # all columns except last
Y = dataset.iloc[:, -1]    # last column

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on train data: ", training_accuracy)

X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on test data: ", test_accuracy)

input_data = (0.1088,0.1278,0.0926,0.1234,0.1276,0.1731,0.1948,0.4262,0.6828,0.5761,0.4733,0.2362,0.1023,0.2904,0.4713,0.4659,0.1415,0.0849,0.3257,0.9007,0.9312,0.4856,0.1346,0.1604,0.2737,0.5609,0.3654,0.6139,0.5470,0.8474,0.5638,0.5443,0.5086,0.6253,0.8497,0.8406,0.8420,0.9136,0.7713,0.4882,0.3724,0.4469,0.4586,0.4491,0.5616,0.4305,0.0945,0.0794,0.0274,0.0154,0.0140,0.0455,0.0213,0.0082,0.0124,0.0167,0.0103,0.0205,0.0178,0.0187,)

input_data_array = np.array(input_data)

#reshape the numpy array

input_data_reshape = input_data_array.reshape(1, -1)

prediction = model.predict(input_data_reshape)

print(prediction)

if (prediction[0] == 'R'):
    print("The object is Rock")
elif (prediction[0] == 'M'):
    print("The object is Mine")
