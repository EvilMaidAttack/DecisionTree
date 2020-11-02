import numpy as np
import pandas as pd
import sys


def make_columns_astype(df, features, as_type):
    for feature in features:
        df[feature] = df[feature].astype(as_type)
    return df


def get_p(T, c, verbose=False):
    """
    Given a set of training objects T with either categorical or continuously valued attributes, calculate the relative
    frequencies p for all values in class c.
    T is a dataframe, c is a string
    """
    size_T = len(T)

    group = T.groupby(c)[c]
    frequency = group.count()
    classes = group.groups

    p = {str(c_i): f_i/size_T for c_i, f_i in zip(classes, frequency)}

    if verbose:
        print(
            f'We have this classes: {list(classes.keys())} and {size_T} datapoints\n')
        print(f'We have following frequencies:\n {frequency}\n')
        print(f'We have found following relative frequencies: \n {p}')

    return p


def entropy(T, c, verbose=False):
    p = np.array(list(get_p(T, c, verbose).values()))
    if 0 in p:
        return 0
    else:
        return -np.sum(p*np.log2(p))


def gini(T, c, verbose=False):
    pass


def misclassification_error(T, c, verbose=False):
    pass


def get_split_canditates(T, A):
    # check type of Attribute A (category or numerical)
    if T[A].dtype.name == 'category':
        print('It\'s a Category')
    if T[A].dtype.name == 'int64' or T[A].dtype.name == 'float64':
        print('It\'s a Numerical')


def make_split(T, A, criterion):
    # check type of Attribute A (category or numerical)
    if T[A].dtype.name == 'category':
        print('It\'s a Category')
    if T[A].dtype.name == 'int64' or T[A].dtype.name == 'float64':
        print('It\'s a Numerical')


def make_best_split(T, A, criterion='information_gain'):
    # strategy:
    #   1. calculate splitpoints (all categories for category or mean between 2 values for numerical)
    #   2. for every splitpoint calculate splitsets where attr_value == splitpoint or attr < splitpoint
    #   3. calculate criterion and keep track of best_criterion_value and best_splitset in regards of splitpoints
    #   4. return the best_criterion_value and best_splitset

    # setup criterion
    try:
        if criterion == 'information_gain':
            criterion = information_gain
        elif criterion == 'gini':
            criterion = gini
        elif criterion == 'misclassification error':
            criterion = misclassification_error
        else:
            raise KeyError(
                'criterion invalid. Please choose from "information_gain", "gini" or "misclassification error".')
    except KeyError as err:
        print('Error caught:', err)
        sys.exit(1)
    # 1.
    split_candidate_values = get_split_canditates(T, A)

    # 2.
    # Todo: pick start values, different best-values for different criterions ?
    best_splitset, best_criterion_value = None, 0
    for split_value in split_candidate_values:
        splitset, criterion_value = make_split(T, A, criterion=criterion)


def information_gain(T, c, A, verbose=False):
    pass


if __name__ == '__main__':
    # import and preprocess train_set
    train_set = pd.read_csv('res/titanic/train.csv')
    test_set = pd.read_csv('res/titanic/test.csv')

    categories = ['Survived', 'Pclass', 'Sex', 'Embarked']
    train_set = make_columns_astype(train_set, categories, 'category')

    train_set['Age'].fillna(round(train_set['Age'].mean()), inplace=True)
    train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)

    # Is this required?
    #features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    #train_set = train_set[features]

    train_set.info()
    make_best_split(train_set, 'SibSp', criterion='information_gain')

    # build the decision tree
    # TODO
