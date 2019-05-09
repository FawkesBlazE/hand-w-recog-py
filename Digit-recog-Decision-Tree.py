#Decision Tree Classifiers
"""'Author: Akash Karar

First Machine Learning Program

Handwritten Digit Reconition using Decision Tree Classifiers

'"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier #Python Library for Decision tree classifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('digit-recognizer\\train.csv').as_matrix()  #Import the Training data set containing 42000 test data

clf = DecisionTreeClassifier()

## Divide the data set into two parts training and testing
train = data[0:21000,1:]
train_label = data[0:21000,0]

##train the classifier
clf.fit(train,train_label)

##testing sample
test = data[21000:,1:]
actual_label = data[21000:,0]

##test one example
#d= test[0]
#print(d)
#d.shape=(28,28)
#print(d)

#plt.imshow(255-d,cmap='gray')
#plt.show()

##test the accuracy of the classifier
predictions = clf.predict(test)

print(predictions)

print(accuracy_score (actual_label,predictions))
