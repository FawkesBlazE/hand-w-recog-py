#K Nearest neighbors Longest Time, SELF CODED
"""'Author: Akash Karar

Third Machine Learning Program

Handwritten Digit Reconition using K Nearest neighbors coded personally to dig deeper

Time: 1hr
'"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier #Python Library for Decision tree classifier
#from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit (self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def predict(self,Xtest):
        predictions=[]
        cont=0
        for row in Xtest:
            label = self.closest(row)
            #label = random.choice(self.Ytrain)
            predictions.append(label)
            print(count+=1)
        return predictions

    def closest(self, row):
        best_dist = euc(row,self.Xtrain[0])
        best_index = 0
        for i in range (1, len(self.Xtrain)):
            dist = euc(row, self.Xtrain[i])
            if dist < best_dist:
                best_dist =dist
                best_index = i
        return self.Ytrain[best_index]


data = pd.read_csv('digit-recognizer\\train.csv').as_matrix()  #Import the Training data set containing 42000 test data

#knn = KNeighborsClassifier()
knn = ScrappyKNN()

## Divide the data set into two parts training and testing
train = data[0:21000,1:]
train_label = data[0:21000,0]

##train the classifier
knn.fit(train,train_label)

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
predictions = knn.predict(test)

print(predictions)

print(accuracy_score (actual_label,predictions))
#count=0
#for i in range(0,21000):
#    count+=1 if predictions[i]==actual_label[i] else 0
#print((count/2100)*100)