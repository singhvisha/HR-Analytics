# Importing necessary library

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

plt.style.use('bmh')

df = pd.read_csv('HR_comma_sep.csv', sep=',')
df.head()
df.info()

groupBy_left = df.groupby('left').mean();
groupBy_left

fig, axis = plt.subplots(nrows=2, ncols=1, figsize =(12,10))
sns.countplot(x = 'salary', hue='left', data = df, ax = axis[0])
sns.countplot(x = 'Department', hue ='left', data = df, ax = axis[1])


# From the data analysis so far we can conclude that 
# we will use following variables as dependant variables in our model
#**Satisfaction Level**
#**Average Monthly Hours**
#**Promotion Last 5 Years**
#**Salary**


subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years','salary']]
subdf.head()

dummies = pd.get_dummies(subdf['salary'])
dummies.head()

dffinal = pd.concat([subdf, dummies], axis=1)
dffinal.head()

X=dffinal.drop('salary',axis='columns')
Y = df['left']
X.head(3)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)

model.coef_

model.predict(X_test)

model.score(X_test,Y_test)





