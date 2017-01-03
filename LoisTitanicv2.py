# LoisTitanic.py
#
#
# LKS, December 2016
#
#
# imports
import pandas as pd
import numpy as np


def testForest(imputation = 'no', strategy=''):
      # load in file into a pandas dataframe
      df=pd.read_csv('Titanic.csv')
      
      #
      # first, let's clean up our data set
      df=df.drop([ 'Sex', 'Name', 'Unnamed: 0'], axis=1)
      
      if imputation == 'no':
          df=df.dropna()
      
      # then let's put the solution off to the side
      # and drop the labeled 'male/female' column in favor of the
      # sex code one (female = 1, male = 0)
      SolutionCol = df['Survived']
      df=df.drop(['Survived'], axis=1)
      
      
      #
      # then we have to fix PClass from 1st, 2nd, etc. to just 1, 2, 3
      # once again, we need numerical values 
      df['PClass'] = df['PClass'].map(lambda x: str(x)[0])
      
      #
      # now split up train-test data
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(
           df, SolutionCol, test_size=0.25)
      
      
      from sklearn.preprocessing import Imputer
      # for imputation
      if imputation == 'yes':
          imp = Imputer(missing_values='NaN', strategy=strategy, axis=0)
          X_train = X_train.apply(pd.to_numeric, errors='coerce')
          X_test = X_test.apply(pd.to_numeric, errors='coerce')
          X_train=imp.fit_transform(X_train)
          X_test=imp.fit_transform(X_test)

      # in the case where no imputation     
      if imputation == 'no':         
          X_train = X_train.apply(pd.to_numeric, errors='coerce')
          X_test = X_test.apply(pd.to_numeric, errors='coerce')
       
      
      
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
      return(acc, precision)




def comparisonPlot(acc, precision, title):
# make plots to compare accuracy and precision of the different imputation (and non imputation)
# methods 
     import matplotlib.pyplot as plt
     fig=plt.figure()
     ax=fig.add_subplot(111)
     ax.plot(precisionTest, lw=2, c='blue')
     ax.yaxis.label.set_color('blue')
     ax.set_xlabel('Test Number')
     ax.set_ylabel('Precision')
     ax2=ax.twinx()
     ax2.plot(accuracyTest, lw=2, c='r')
     ax2.set_ylabel('Accuracy')
     ax2.yaxis.label.set_color('red')
     
     plt.title('Median Precision: '+str(int(np.nanmedian(precisionTest)*100)/100.0) +\
               '+/- '+str(int(100*np.std(precisionTest))/100.) +', Median Accuracy: '+ \
               str(int(np.nanmedian(accuracyTest)*100)/100.)+'+/- '+str(int(100*np.std(precisionTest))/100.))
                                       
     plt.savefig(title)

# generate the tests, will produce figures
testsImp=['yes', 'yes', 'yes', 'no']
testsStrat=['mean', 'median', 'most_frequent', '']
for item in range(len(testsImp)):
    precisionTest=np.zeros(100)
    accuracyTest=np.zeros(100)
    for ii in range(100):
        TF=testForest(imputation=testsImp[item],strategy=testsStrat[item])
        precisionTest[ii]=TF[0]
        accuracyTest[ii] = TF[1]
    comparisonPlot(accuracyTest, precisionTest, 'AccVsPrecisionImp='+testsImp[item]+\
                   'strat='+testsStrat[item]+'.png')

#
# Now let's make a Random Forest Model Adding more features 
#
def testForestAddFeatures():
      # load in file into a pandas dataframe
      df=pd.read_csv('Titanic.csv')
      
      #
      # first, let's clean up our data set
      df=df.dropna()
      df['NameLength']=df['Name'].map(lambda x: len(x))
      #
      # remove AgeDecade after playing around a bit more for a slightly better result. 
      df['AgeDecade']=df['Age'].map(lambda x: int(x/10.))   
      #
      # get Mr, Mrs, Miss, Colonel, etc.
      char1=', '; char2=' ' # based on how the excel table sets up the names 
      df['prefix']=df['Name'].map(lambda mystr: mystr[mystr.find(char1)+1 : mystr[(mystr.find(char1)+2):].find(char2)+mystr.find(char1)+2])

      # Note, there are bad prefixes (i.e. No prefix, just name) However, it is less than 1% of our dataset and very difficult to
      # automatically distinguish between Lady, Madame, etc. which also only appear once or twice in our entire data set. We
      # will leave them
      from sklearn import preprocessing
      le = preprocessing.LabelEncoder()
      le.fit(df['prefix'])
      df['prefixNumber']=le.transform(df['prefix']) 
      df=df.drop([ 'Sex', 'Name', 'Unnamed: 0', 'prefix', 'PClass'], axis=1)
            
      # then let's put the solution off to the side
      # and drop the labeled 'male/female' column in favor of the
      # sex code one (female = 1, male = 0)
      SolutionCol = df['Survived']
      df=df.drop(['Survived'], axis=1)
      
      #
      # then we have to fix PClass from 1st, 2nd, etc. to just 1, 2, 3
      # once again, we need numerical values 
      #df['PClass'] = df['PClass'].map(lambda x: str(x)[0])
      
      #
      # now split up train-test data
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(
           df, SolutionCol, test_size=0.25, random_state=1)      
      
      X_train = X_train.apply(pd.to_numeric, errors='coerce')
      X_test = X_test.apply(pd.to_numeric, errors='coerce')
           
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
      print(list(zip(df.columns.values, clf.feature_importances_)))
      return(acc, precision)

#
# test the new model with additional features, play around with this function and see what you get :) 
testForestAddFeatures()
