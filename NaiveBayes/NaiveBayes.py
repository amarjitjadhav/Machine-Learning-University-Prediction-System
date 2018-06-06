import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas
import math
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


class project(object):
    def loadData(self):
        iData = pandas.read_csv('NaiveByes/ASU1.csv')
        data = pandas.DataFrame(data=iData)
        # data = preprocessing.scale(iData[:, 0:-1])
        data1 = data.ix[:, [4, 7, 8, 9, 10, 11, 13, 14, 18]]

        label = data1.ix[:, [0]]

        label["Result"] = np.where(label["Result"] == "Accept", 1, 0)

        trainingData = data1.ix[:, 1:9]
        # trainingData.ix[:,7] = preprocessing.scale(trainingData.ix[:,7])

        test = pandas.read_csv('NaiveByes/test.csv')
        testingData = pandas.DataFrame(data=test)

        return trainingData, label, testingData

    # end loadData

    def calGNB(self, trainingData, label, testingData):
        gnb = GaussianNB()
        gnb.fit(trainingData.values, label)
        y_pred = gnb.predict(testingData)

        print(y_pred)


# end calGNB

# end class project

def main():
    object1 = project()

    trainingData, label, testingData = object1.loadData()

    object1.calGNB(trainingData, label, testingData)


# end main


if __name__ == '__main__':
    main()