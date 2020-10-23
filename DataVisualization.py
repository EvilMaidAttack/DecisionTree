import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('res/titanic/train.csv')

age_distribution_0 = df[df['Survived']==0]['Parch'].dropna()
age_distribution_1 = df[df['Survived']==1]['Parch'].dropna()


fig = plt.figure()
ax_0 = fig.add_subplot(211)
ax_1 = fig.add_subplot(212)

ax_0.hist(age_distribution_0, color='red')
ax_1.hist(age_distribution_1, color='green')

fig.suptitle('Age distribution of survivors (green) and deceased (red)')
ax_0.set_xlabel('Age')
ax_0.set_ylabel('Cases')
fig.suptitle('Age distribution of survivors (green) and deceased (red)')
ax_0.set_xlabel('Age')
ax_0.set_ylabel('Cases')
ax_0.set_title('People who deceased')
ax_1.set_xlabel('Age')
ax_1.set_ylabel('Cases')
ax_1.set_title('People who survived')



plt.show()
