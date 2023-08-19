# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 01:23:21 2020

@author: Administrator
"""

#####import the data
import pandas as pd
import os
path = '/Users/xiaojing/5thSemester/Datawarehouse/project'
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
data_project = pd.read_csv(fullpath)

### Basic info of the dataset
data_project .shape
### find the number of columns
cols=len(data_project .axes[1])
print("Number of Columns: "+str(cols))
data_project .columns.values
### Finding the number of rows
rows=len(data_project .axes[0])
print("Number of Rows: "+str(rows))
data_project.describe()
data_project.dtypes 

#####Data Modeling & Warangling
##get the unique values for selecting the attributes
print(data_project['Status'].unique())

print(data_project['Primary_Offence'].unique())
print(data_project['Occurrence_Date'].unique())
print(data_project['Occurrence_Year'].unique())
print(data_project['Occurrence_Month'].unique())
print(data_project['Occurrence_Day'].unique())
print(data_project['Location_Type'].unique())
print(data_project['Division'].unique())
print(data_project['City'].unique())
print(data_project['Location_Type'].unique())
print(data_project['Bike_Type'].unique())
print(data_project['Bike_Make'].unique())
print(data_project['Bike_Model'].unique())
print(data_project['Bike_Speed'].unique())
print(data_project['Cost_of_Bike'].unique())
print(data_project['Bike_Colour'].unique())
print(data_project['Hood_ID'].unique())
print(data_project['Neighbourhood'].unique())

###find the relationship in dataset
import seaborn as sns
sns.distplot(data_project['Cost_of_Bike'])
# Check all correlations
sns.pairplot(data_project)
# find the relationship between status and cost_of_bike
data_project['Recovered'] = [1 if Status=='RECOVERED' else 0 for Status in data_project.Status]
sub=data_project[['Recovered','Cost_of_Bike']]
sns.pairplot(sub)
###data normalization
sub_norm = (sub - sub.min()) / (sub.max() - sub.min())
sns.lineplot(data=sub_norm)
#####calculate coreelation
recovered=data_project['Recovered']
cost=data_project['Cost_of_Bike']
recovered.corr(cost) 
#####weak positive correlation

###data slicing
subdata_project =data_project[['Premise_Type','Bike_Type','Cost_of_Bike','Status','Neighbourhood']]
###removing 0.0 from cost of bike
df_project =subdata_project.drop(subdata_project[data_project.Cost_of_Bike == 0.0].index)
### drop rows where Status = Unknown
df_project = df_project .drop(df_project [df_project.Status == 'UNKNOWN'].index)

### Handling missing values
df_project['Cost_of_Bike'].fillna(df_project['Cost_of_Bike'].median(),inplace=True)
###find out null values
print(df_project.isnull().sum())
###overall
df_project.describe()
df_project.info()

#####balance the data
df_project['balance'] = [1 if Status=='STOLEN' else 0 for Status in df_project.Status]
df_project['balance'].value_counts()
##balance the original dataset by Down-sample Majority Class
from sklearn.utils import resample
# Separate majority and minority classes
df_majority = df_project[df_project.balance==1]
df_minority = df_project[df_project.balance==0]
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=19956,    #set the number of majority
                                 random_state=123) # reproducible results
##got the balanced data set
df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced['balance'].value_counts()
##overall
df_balanced = df_balanced.drop('balance', axis=1)
df_balanced.shape
df_balanced.dtypes 
df_balanced.info()

###scale the attributes
sub_df_balanced=df_balanced.drop('Status',axis=1)
##Convert the categorical values into numeric columns using the get dummies
categoricals = []
for col, col_type in sub_df_balanced.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)     
print(categoricals)
df_ohe = pd.get_dummies(sub_df_balanced, columns=categoricals, dummy_na=False)
print(df_ohe.head())
print(df_ohe.columns.values)
print(len(df_ohe) - df_ohe.count())
from sklearn import preprocessing
# Get column names first
names = df_ohe.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_ohe)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df.head())
print(scaled_df.dtypes)
scaled_df.info
print(scaled_df.columns.values)

#####Prediction Models

##deviding data
# Separate input features (X) and target variable (y)
x = scaled_df
y = df_balanced['Status']
# Split the data into train test
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)


###Logistic Regression
from sklearn.linear_model import LogisticRegression
import numpy as np 
dependent_variable = 'Status'
# Another way to split the features
x = scaled_df[scaled_df.columns.difference([dependent_variable])]
x.dtypes
y = df_balanced[dependent_variable]
# Split the data into train test
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)
#build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)
# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)

testY_predict = lr.predict(testX)
#print(testY_predict)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))

###Visualize the Confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt     
cm = confusion_matrix(testY, testY_predict, labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Stolen', 'Recovered']); ax.yaxis.set_ticklabels(['Stolen', 'Recovered']);
plt.show()


###Decision Tree

#build the tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np 
for i in range(25,35):
    dt1_project = DecisionTreeClassifier(criterion='entropy',max_depth=i, min_samples_split=20, random_state=123)
    dt1_project.fit(trainX,trainY)
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    score = np.mean(cross_val_score(dt1_project, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
    print(score)
    
    
##when i=34 the score is highest
dt1_project = DecisionTreeClassifier(criterion='entropy',max_depth=34, min_samples_split=20, random_state=123)
dt1_project.fit(trainX,trainY)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
score = np.mean(cross_val_score(dt1_project, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print(score)   
 

from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

importance = dt1_project.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


### Test the model using the testing data
testY_predict = dt1_project.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY,testY_predict,labels))

###Visualize the Confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt     
cm = confusion_matrix(testY, testY_predict, labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Stolen', 'Recovered']); ax.yaxis.set_ticklabels(['Stolen', 'Recovered']);
plt.show()


######Deploy the model file
import joblib 
joblib.dump(dt1_project, '/Users/xiaojing/5thSemester/Datawarehouse/project/projectmodel_dt.pkl')
print("Model dumped!")

model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, '/Users/xiaojing/5thSemester/Datawarehouse/project/projectmodel_columns.pkl')
print("Models columns dumped!")

























