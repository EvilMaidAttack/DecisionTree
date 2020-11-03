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


def misclassification(T, c, verbose=False):
    pass


def make_split(T, A):
    T_subs = {}
    if T[A].dtype.name == 'category':
        group = T.groupby(A)
        for category_value in group.groups:
            T_sub = T[T[A] == category_value]
            T_subs[category_value] = T_sub

    if T[A].dtype.name == 'int64' or T[A].dtype.name == 'float64':
        print('It\'s a Numerical')

    return T_subs


def make_best_split(T, As, criterion='information_gain', verbose=False):
    # strategy for all A in As:
    #   1. calculate splitpoints (all categories for category or mean between 2 values for numerical)
    #   2. for every splitpoint calculate splitsets where attr_value == splitpoint or attr < splitpoint
    #   3. calculate criterion and keep track of best_criterion_value and best_splitset in regards of the As
    #   4. return the best_criterion_value and best_splitset

    # check type of Attribute A (category or numerical)
    # setup criterion
    try:
        if criterion == 'entropy':
            criterion = information_gain
        elif criterion == 'gini':
            criterion = gini_index
        elif criterion == 'misclassification':
            criterion = misclassification_error
        else:
            raise KeyError(
                'criterion key word invalid. Please choose from "entropy", "gini" or "misclassification".')
    except KeyError as err:
        print('Error caught:', err)
        sys.exit(1)

    best_criterion_value, best_T_subs = None, None
    for A in As:
        T_subs = make_split(T, A)
        criterion_value = criterion(T, A, T_subs, verbose=verbose)
        # catch the start phase and initialize best values
        if best_criterion_value == None:
            best_criterion_value = criterion_value
            best_T_subs = T_subs
        # because information gain is better when it has a higher value (unlike the other 2 criterions)
        elif criterion.__name__ == 'information_gain':
            if criterion_value > best_criterion_value:
                best_criterion_value = criterion_value
                best_T_subs = T_subs
        # case for gini index and misclassification error. Lower value is better
        else:
            if criterion_value < best_criterion_value:
                best_criterion_value = criterion_value
                best_T_subs = T_subs

    return best_T_subs, best_criterion_value


def information_gain(T, c, T_subs, verbose=False):
    entropy_before_split = entropy(T, c, verbose=verbose)
    entropies_after_split = [entropy(T_sub, c, verbose)
                             for T_sub in T_subs.values()]
    lengths_T_sub = [len(T_sub.index) for T_sub in T_subs.values()]
    print('lengths: ', lengths_T_sub)
    print('entropy_before: ', entropy_before_split)
    print('entropy_after: ', entropies_after_split)


def gini_index(T, c, T_subs, verbose=False):
    pass


def misclassification_error(T, c, T_subs, verbose=False):
    pass


if __name__ == '__main__':
    # import and preprocess train_set
    train_set = pd.read_csv('res/titanic/train.csv')
    test_set = pd.read_csv('res/titanic/test.csv')

    categories = ['Survived', 'Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']
    train_set = make_columns_astype(train_set, categories, 'category')

    train_set['Age'].fillna(round(train_set['Age'].mean()), inplace=True)
    train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)

    features = ['Survived', 'Pclass', 'Sex', 'Age',
                'SibSp', 'Parch', 'Fare', 'Embarked']

    train_set.info()
    T_subs, criterion_value = make_best_split(
        train_set, categories, criterion='entropy', verbose=True)

    # build the decision tree
    # TODO
