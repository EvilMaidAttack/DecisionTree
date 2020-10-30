import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

traindata = pd.read_csv('titanic/train.csv')
testdata = pd.read_csv('titanic/test.csv')
features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(traindata[features])
X_test = pd.get_dummies(testdata[features])
y = traindata['Survived']

forest = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=1)
forest.fit(X, y)

predictions = forest.predict(X_test)
print(predictions)

output = pd.DataFrame(
    {'PassengerId': testdata.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
