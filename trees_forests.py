import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def get_data(train_file, test_file):
    train_data = pd.read_csv('data/train/' + train_file)
    test_data = pd.read_csv('data/test/' + test_file)
    uni = train_data['UniversityApplied'][0]
    X_train = train_data[['GRE', 'GRE-V', 'GRE (Quants)', 'AWA', 'TOEFL',
                          'Work-Ex', 'International Papers', 'Percentage']]
    y_train = pd.get_dummies(train_data['Result'])['Accept']
    X_test = test_data.drop(['Result'], axis=1)
    y_test = pd.get_dummies(test_data['Result'])['Accept']
    return uni, X_train, y_train, X_test, y_test


def classifier(clf, train_List, test_list):
    accuracies = []
    all_X_train, all_y_train = pd.DataFrame(), pd.Series()
    all_X_test, all_y_test = pd.DataFrame(), pd.Series()
    for files in zip(train_List, test_list):
        uni, X_train, y_train, X_test, y_test = get_data(*files)
        clf.fit(X_train, y_train)
        for _ in range(10):
            acc = clf.score(X_test, y_test)
            accuracies.append(acc)
        print(uni, ': ', str(round(np.mean(acc)*100, 2)), '% accuracy', sep='')
        all_X_train = all_X_train.append(X_train)
        all_y_train = all_y_train.append(y_train)
        all_X_test = all_X_test.append(X_test)
        all_y_test = all_y_test.append(y_test)
    accuracies = []
    clf.fit(all_X_train, all_y_train)
    for _ in range(10):
        acc = clf.score(all_X_test, all_y_test)
        accuracies.append(acc)
    print('All universities together: ',
          str(round(np.mean(acc)*100, 2)), '% accuracy', sep='')


def main():
    train_List = ['asu.csv', 'clemson.csv', 'iitc.csv', 'mtu.csv']
    test_list = ['asu_test.csv', 'clemson_test.csv',
                 'iitc_test.csv', 'mtu_test.csv']
    print('Average acuracies, simple decision tree:')
    classifier(DecisionTreeClassifier(), train_List, test_list)
    print('\nAverage accuracies, random forest:')
    classifier(RandomForestClassifier(), train_List, test_list)


if __name__ == '__main__':
    main()
