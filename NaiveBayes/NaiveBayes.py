import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


class Model(object):

    def loadData(self, trainD, testD):
        train_data = pd.read_csv('../data/train/' + trainD)
        test_data = pd.read_csv('../data/test/' + testD)

        # Fetch university name
        uni_name = train_data['UniversityApplied'][0]

        # Select features that are important to model
        X_train = train_data[['GRE', 'GRE-V', 'GRE (Quants)', 'AWA', 'TOEFL',
                              'Work-Ex', 'International Papers', 'Percentage']]

        # create training Label
        y_train = pd.get_dummies(train_data['Result'])['Accept']

        # create testing data
        X_test = test_data.drop(['Result'], axis=1)

        # create testing Label
        y_test = pd.get_dummies(test_data['Result'])['Accept']

        # Calculate prediction
        predictions, _ = self.calGNB(X_train, y_train, X_test)

        # confusionMatrix = metrics.confusion_matrix(y_test, predictions)
        # print('Confusion matrix:')
        # print(confusionMatrix)

        # Calculate accuracy
        result = accuracy_score(y_test, predictions)

        return result, uni_name

    # end loadData

    def calGNB(self, X_train, y_train, X_test):
        # feed data to gaussian classifier
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)

        # gives result in either 0 or 1
        predictions = gnb.predict(X_test)

        # gives result in probability
        y_probs = gnb.predict_proba(X_test)

        return predictions, y_probs[:,1]


# end calGNB

# end class project


def main():
    model = Model()

    university_names = []
    accuracy_score = []

    # list of training csv files
    training_List = ['asu.csv', 'clemson.csv', 'iitc.csv', 'mtu.csv']
    # list of testing csv files
    testing_list = ['asu_test.csv', 'clemson_test.csv',
                    'iitc_test.csv', 'mtu_test.csv']

    # get results for each university data
    for trainD, testD in zip(training_List, testing_list):
        result, uni_name = model.loadData(trainD, testD)
        accuracy_score.append(result * 100)
        university_names.append(uni_name)
    # end for

    print('University predictions for student:')
    for uni, accuracy in zip(university_names, accuracy_score):
        print(uni + ' -> ' + str(accuracy) + '%\n')


# end for

# end main


if __name__ == '__main__':
    main()
