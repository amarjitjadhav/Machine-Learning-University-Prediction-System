import numpy as np
import pandas
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


class project(object):

    def loadData(self, trainD, testD):
        train_data = pandas.read_csv('../data/train/'+trainD)

        # Fetch university name
        uni_name = train_data['UniversityApplied'][0]

        # Select features that are important to model
        X_train = train_data.ix[:, [7, 8, 9, 10, 11, 13, 14, 18]]

        # create training Label
        y_train = train_data['Result']
        y_train = np.where(y_train == "Accept", 1, 0)

        # create training data

        test_data = pandas.read_csv('../data/test/'+testD)

        # create testing Label
        y_test = test_data['Result']
        y_test = np.where(y_test == "Accept", 1, 0)

        # create testing data
        X_test = test_data.ix[:, 1:9]

        # Calculate prediction and probability
        predictions, probability = self.calGNB(X_train, y_train, X_test)

        # confusionMatrix = metrics.confusion_matrix(y_test, predictions)
        # print("Confusion matrix:")
        # print(confusionMatrix)

        # Calculate accuracy
        result = accuracy_score(y_test, predictions)

        return result, uni_name

    # end loadData

    def calGNB(self, trainingData, label, X_test):
        # feed data to gaussian classifier
        gnb = GaussianNB()
        gnb.fit(trainingData.values, label)

        # gives result in either 0 or 1
        y_result = gnb.predict(X_test)

        # gives result in probability
        y_prob = gnb.predict_proba(X_test)

        return y_result, y_prob[:, 1]


# end calGNB

# end class project


def main():
    object1 = project()

    university_Names = []
    accuracy_score = []

    # list of training csv files
    training_List = ['asu.csv', 'clemson.csv', 'iitc.csv', 'mtu.csv']
    # list of testing csv files
    testing_list = ['asu_test.csv', 'clemson_test.csv', 'iitc_test.csv', 'mtu_test.csv']

    # get results for each university data
    for trainD, testD in zip(training_List, testing_list):
        result, uni_name = object1.loadData(trainD, testD)
        accuracy_score.append(result * 100)
        university_Names.append(uni_name)
    # end for

    print("University predictions for student:")
    for uni, accuracy in zip(university_Names, accuracy_score):
        print(str(uni) + " -> " + str(accuracy) + "%")
        print("")


# end for

# end main


if __name__ == '__main__':
    main()
