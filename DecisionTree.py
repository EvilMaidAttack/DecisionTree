import numpy as np
import pandas as pd

def make_columns_astype(df, features, as_type):
    for feature in features:
        df[feature] = df[feature].astype(as_type)
    return df

def get_p(T,c, verbose=False):
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
        print(f'We have this classes: {list(classes.keys())} and {size_T} datapoints\n')
        print(f'We have following frequencies:\n {frequency}\n')
        print(f'We have found following relative frequencies: \n {p}')

    return p


def entropy(T, c, verbose=False):
    p = np.array(list(get_p(T, c, verbose).values()))
    if 0 in p:
        return 0
    else:
        return -np.sum(p*np.log2(p))

def make_split(T,A,criterion='entropy'):
    pass


# Define this function
def information_gain(T,c,A,verbose=False):
    pass



if __name__=='__main__':
    # import and preprocess train_set
    train_set = pd.read_csv('res/titanic/train.csv')

    categories = ['Survived', 'Pclass', 'Sex','Embarked']
    train_set = make_columns_astype(train_set, categories, 'category')

    train_set['Age'].fillna(round(train_set['Age'].mean()), inplace=True)
    train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)

    # Is this required?
    #features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    #train_set = train_set[features]

    train_set.info()


    # build the decision tree
    # TODO










