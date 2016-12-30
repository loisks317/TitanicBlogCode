# LoisTitanic.py
#
#
# LKS, December 2016
#
#
# imports
import pandas as pd
import numpy as np

# load in file into a pandas dataframe
df=pd.read_csv('Titanic.csv')

#
# first, let's clean up our data set
df=df.dropna()

# then let's put the solution off to the side
# and drop the labeled 'male/female' column in favor of the
# sex code one (female = 1, male = 0)
SolutionCol = df['Survived']
df=df.drop(['Survived', 'Sex', 'Name', 'Unnamed: 0'], axis=1)

#
# then we have to fix PClass from 1st, 2nd, etc. to just 1, 2, 3
# once again, we need numerical values 
df['PClass'] = df['PClass'].map(lambda x: str(x)[0])

#
# now split up train-test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     df, SolutionCol, test_size=0.25, random_state=2)

# setup the model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions=clf.predict(X_test)

# score the model
# how accurate was the model overall
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, predictions)

# what was it's precision
from sklearn.metrics import precision_score
precision=precision_score(y_test, predictions)

# what is the TP, FP, FN, TN rate? (true positive, false positive, etc.) 
from sklearn.metrics import confusion_matrix
confusionMatrix=confusion_matrix(y_test, predictions)

# print out the values 
print('accuracy is: ' + str(int(acc*100)) + '%')
print('precision is: ' + str(precision))
print ('confusion matrix: '+ str(confusion_matrix(y_test, predictions)))

# what were the most important features?
list(zip(df.columns.values, clf.feature_importances_))

