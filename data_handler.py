import pandas as pd


def train_test_split(train_file, test_file):
    train_data = pd.read_csv('data/train/' + train_file)
    test_data = pd.read_csv('data/test/' + test_file)
    uni = train_data['UniversityApplied'][0]
    X_train = train_data[['GRE', 'GRE-V', 'GRE (Quants)', 'AWA', 'TOEFL',
                          'Work-Ex', 'International Papers', 'Percentage']]
    y_train = pd.get_dummies(train_data['Result'])['Accept']
    X_test = test_data.drop(['Result'], axis=1)
    y_test = pd.get_dummies(test_data['Result'])['Accept']
    return uni, X_train, X_test, y_train, y_test
