from data_handler import train_test_split

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def classifier(clf, train_list, test_list):
    accuracies = []
    all_X_train, all_y_train = pd.DataFrame(), pd.Series()
    all_X_test, all_y_test = pd.DataFrame(), pd.Series()
    for files in zip(train_list, test_list):
        uni, X_train, X_test, y_train, y_test = train_test_split(*files)
        for _ in range(10):
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            accuracies.append(acc)
        avg_acc = np.mean(accuracies)*100
        print(uni, ': ', str(round(avg_acc, 2)), '% accuracy', sep='')
        all_X_train = all_X_train.append(X_train)
        all_y_train = all_y_train.append(y_train)
        all_X_test = all_X_test.append(X_test)
        all_y_test = all_y_test.append(y_test)
    accuracies = []
    for _ in range(10):
        clf.fit(all_X_train, all_y_train)
        acc = clf.score(all_X_test, all_y_test)
        accuracies.append(acc)
    avg_acc = np.mean(accuracies)*100
    print('All universities together: ',
          str(round(avg_acc, 2)), '% accuracy', sep='')


def main():
    train_list = ['asu.csv', 'clemson.csv', 'iitc.csv', 'mtu.csv']
    test_list = ['asu_test.csv', 'clemson_test.csv',
                 'iitc_test.csv', 'mtu_test.csv']
    print('Average acuracies, simple decision tree:')
    classifier(DecisionTreeClassifier(), train_list, test_list)
    print('\nAverage accuracies, random forest:')
    classifier(RandomForestClassifier(), train_list, test_list)


if __name__ == '__main__':
    main()
