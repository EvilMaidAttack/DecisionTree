import numpy as np
import pandas as pd
import sys
from copy import deepcopy
from DecisionTreeNode import DecisionTreeNode


class DecisionTree():

    def __init__(self, criterion_string, attributes, nodes=[]):
        """Decision Tree classifier for the Titanic dataset to predict if person survives.

        Args:
            criterion_string (string): String to indicate which impurity measurements to use. Supports 'information gain', 'gini index' and 'misclassification error'
            nodes (list, optional): List of DecisionTreeNode representing the Nodes in the DecisionTree. Defaults to [].
        """
        super().__init__()
        self.nodes = nodes
        self.criterion_string = criterion_string
        self.attributes = attributes

    def fit(self, X):
        attributes = self.attributes
        # TODO: finish this

    def predict(self, X):
        pass


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


def information_gain(T, c, T_subs, verbose=False):
    # calculate the information gain. Best value = 1 and worst value = 0.
    # T is the parent dframe
    # c is a string representing the class (here specifically "Survived"),
    # T_subs are the sub-dframes after split as a dictionary of dframes with attribute value as 'keys'
    if verbose:
        print('\nCalculating information gain!')

    entropy_before_split = entropy(T, c, verbose=verbose)
    entropies_after_split = [entropy(T_sub, c, verbose)
                             for T_sub in T_subs.values()]
    length_parent_set = len(T.index)
    lengths_T_sub = [len(T_sub.index) for T_sub in T_subs.values()]

    result = entropy_before_split - \
        np.sum(np.array(lengths_T_sub)/length_parent_set
               * np.array(entropies_after_split))

    if verbose:
        print('Parent set length: ', length_parent_set)
        print('Subset lengths: ', lengths_T_sub)
        print('Entropy parent set: ', entropy_before_split)
        print('entropy subsets: ', entropies_after_split)
        print('--- Information gain: ', result)
        print('Finished calculation of information gain!')

    return result


def gini_index(T, c, T_subs, verbose=False):
    pass


def misclassification_error(T, c, T_subs, verbose=False):
    pass


def get_numerical_split_candidates(T, A):
    # Sort data ascending by value in attribute A. Calculate mean between 2 adjacend values
    # Each mean should be considered as a possible split point. --> many duplicates possible so we drop them
    # returns a dframe with one column 'Mean'
    T_sorted = T.sort_values(A).reset_index(drop=True)
    T_sorted['Mean'] = T_sorted.groupby(
        np.arange(len(T_sorted)) // 2)[A].transform('mean')
    means = T_sorted['Mean'].to_frame().drop_duplicates(ignore_index=True)[1:]
    return means


def find_best_numerical_splitpoint(T, A, means, verbose=False):
    # For a numerical attribute to find the best splitpoint we first calculate the means according to 'get_numerical_splitpoints'
    # For each mean we split the data with '< mean' and '>= mean'
    # We measure the impurity of the split with 'information_gain'
    # We take the splitset and splitpoint with the lowest impurity (highest information gain) as the best split
    best_mean = means[0]
    T_subs_best = {'<': T[T[A] < means[0]], '>=': T[T[A] >= means[0]]}
    T_subs_candidate = {}
    inf_gain_best = information_gain(
        T, 'Survived', T_subs_best, verbose=False)

    for i, mean in enumerate(means[1:]):
        T_subs_candidate['<'] = T[T[A] < mean]
        T_subs_candidate['>='] = T[T[A] >= mean]
        inf_gain_candidate = information_gain(
            T, 'Survived', T_subs_candidate, verbose=False)
        l = [len(T.index) for T in T_subs_candidate.values()]
        if inf_gain_best < inf_gain_candidate:
            candidate_len = [len(T.index)
                             for T in T_subs_candidate.values()]
            best_len = [len(T.index) for T in T_subs_best.values()]
            if verbose:
                print(
                    f'{i+1}/{len(means)}\nold: {inf_gain_best:.4f} with len: {best_len} \nnew: {inf_gain_candidate:.4f} with len: {candidate_len}')
            inf_gain_best = inf_gain_candidate
            T_subs_best = deepcopy(T_subs_candidate)
            best_mean = deepcopy(mean)

    return T_subs_best, best_mean


def make_split(T, A, verbose=False):
    if verbose:
        print(f'\ninitializing make_split  for: {A}')
    T_subs = {}
    if T[A].dtype.name == 'category':
        group = T.groupby(A)
        for category_value in group.groups:
            T_sub = T[T[A] == category_value]
            T_subs[category_value] = T_sub

    if T[A].dtype.name == 'int64' or T[A].dtype.name == 'float64':
        means = np.array(get_numerical_split_candidates(T, A))[:, 0]

        T_subs_best, best_splitpoint = find_best_numerical_splitpoint(
            T, A, means, False)

        if verbose:
            print('Best splitpoint for "{}" found at {:.4f}'.format(
                A, best_splitpoint))

        # Dont forget to deepcopy !!!
        T_subs = deepcopy(T_subs_best)

    if verbose:
        print(f'make_split finished for {A} with keys: {list(T_subs.keys())}')

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

    best_criterion_value, best_T_subs, best_A = None, None, None
    for A in As:
        if verbose:
            print(
                f'\n\n--------------------Analyse for Attribute: {A} ----------------------')
        T_subs = make_split(T, A, verbose)
        criterion_value = criterion(T, 'Survived', T_subs, verbose=verbose)
        # catch the start phase and initialize best values
        if best_criterion_value == None:
            best_criterion_value = criterion_value
            best_T_subs = deepcopy(T_subs)
            best_A = A
        # because information gain is better when it has a higher value (unlike the other 2 criterions)
        if criterion.__name__ == 'information_gain':
            if criterion_value > best_criterion_value:
                best_criterion_value = criterion_value
                best_T_subs = deepcopy(T_subs)
                best_A = A
        # case for gini index and misclassification error. Lower value is better
        else:
            if criterion_value < best_criterion_value:
                best_criterion_value = criterion_value
                best_T_subs = deepcopy(T_subs)
                best_A = A

    if verbose:
        print('\n=============================================================================\n'
              'Found best split with Attribute {} and {} = {:.4}'
              '\n=============================================================================\n'
              .format(best_A, criterion.__name__, best_criterion_value))

    return best_T_subs, best_criterion_value


if __name__ == '__main__':
    # import and preprocess train_set
    train_set = pd.read_csv('res/titanic/train.csv')
    test_set = pd.read_csv('res/titanic/test.csv')

    categories = ['Survived', 'Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']
    train_set = make_columns_astype(train_set, categories, 'category')

    train_set['Age'].fillna(round(train_set['Age'].mean()), inplace=True)
    train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)

    features = ['Pclass', 'Sex', 'Age',
                'SibSp', 'Parch', 'Fare', 'Embarked']

    train_set.info()
    T_subs, criterion_value = make_best_split(
        train_set, features, criterion='entropy', verbose=True)

    # build the decision tree
    # TODO
