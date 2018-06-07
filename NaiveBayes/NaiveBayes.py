import numpy as np
import pandas
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


class project(object):

    def loadData(self, trainD, testD):
        iData = pandas.read_csv(trainD)
        data = pandas.DataFrame(data=iData)

        # Fetch Uuniversity name
        uni_Name = data.ix[0, 2]

        # Select features that are important to model
        data1 = data.ix[:, [4, 7, 8, 9, 10, 11, 13, 14, 18]]

        # create tarining Label
        trainingLabel = data1.ix[:, 0]
        trainingLabel = np.where(trainingLabel == "Accept", 1, 0)

        # create training data
        trainingData = data1.ix[:, 1:9]

        test = pandas.read_csv(testD)
        testingData = pandas.DataFrame(data=test)

        # create testing Label
        testLabel = test.ix[:, 0]
        testLabel = np.where(testLabel == "Accept", 1, 0)

        # create testing data
        testData = test.ix[:, 1:9]

        # Calculate prediction and probability
        predictions, probability = self.calGNB(trainingData, trainingLabel, testData)

        # confusionMatrix = metrics.confusion_matrix(testLabel, predictions)
        # print("Confusion matrix:")
        # print(confusionMatrix)

        # Calculate accuracy
        result = accuracy_score(testLabel, predictions)

        return result, uni_Name

    # end loadData

    def calGNB(self, trainingData, label, testData):
        # feed data to gaussian classifier
        gnb = GaussianNB()
        gnb.fit(trainingData.values, label)

        # gives result in either 0 or 1
        y_result = gnb.predict(testData)

        # gives result in probability
        y_prob = gnb.predict_proba(testData)

        return y_result, y_prob[:, 1]


# end calGNB

# end class project


def main():
    object1 = project()

    university_Names = []
    accuracy_score = []

    # list of training csv files
    training_List = ['ASU1.csv', 'clemson.csv', 'IIT-chicago.csv', 'MTU.csv']
    # list of testing csv files
    testinig_List = ['TestData_ASU.csv', 'TestData_clemson.csv', 'IITC_Test.csv', 'TestData_MTU.csv']

    # get results for each university data
    for trainD, testD in zip(training_List, testinig_List):
        result, uni_Name = object1.loadData(trainD, testD)
        accuracy_score.append(result * 100)
        university_Names.append(uni_Name)
    # end for

    print("University predictions for student:")
    for uni, accuracy in zip(university_Names, accuracy_score):
        print(str(uni) + " -> " + str(accuracy) + "%")
        print("")


# end for

# end main


if __name__ == '__main__':
    main()