# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 01:23:21 2020

@author: Administrator
"""

import pandas as pd
import os
path = '/Users/xiaojing/5thSemester/Datawarehouse/project'
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
data_project = pd.read_csv(fullpath)
##### Basuc info of the dataset
data_project .shape
##### find the number of columns
cols=len(data_project .axes[1])
print("Number of Columns: "+str(cols))
data_project .columns.values
#### Finding the number of rows
rows=len(data_project .axes[0])
print("Number of Rows: "+str(rows))
data_project.describe()
data_project.dtypes 

##### get the mean of cost time     
mean_cost=data_project['Cost_of_Bike'].mean()
print("The mean of cost of bike is: "+ str(mean_cost))

# Fill the missing values with zeros
data_project.fillna(0,inplace=True)

#####find the relationship between month and cost
occurence_month=data_project['Occurrence_Month']
cost=data_project['Cost_of_Bike']
occurence_month.corr(cost) 
#####negative correlation


# Plot a histogram
import matplotlib 
import matplotlib.pyplot as plt
hist_data_project= plt.hist(data_project['Occurrence_Year'],bins=16)
data_project(axis='y', alpha=0.75)
data_project('Year')
data_project('Frequency')
data_project('Frequency of Bicycle Theft by Year')
# Plot a boxplot
import matplotlib.pyplot as plt
plt.boxplot(data_project ['Occurrence_Year'])
plt.ylabel('Occurrence_Year')
plt.title('Box Plot of Occurrence Year')

#data slicing
subdata_project =data_project[['Location_Type','Premise_Type','Bike_Type','Cost_of_Bike','Status','Neighbourhood']]

#get the attributes in [status]
print(subdata_project['Status'].unique())
#finf out null values
print(subdata_project.isnull().sum())
subdata_project.dropna(axis=0,how='any',inplace=True) 
subdata_project.info() 
## drop rows where Status = Unknown
df_project = subdata_project.drop(subdata_project[subdata_project.Status == 'UNKNOWN'].index)
df_project.shape
df_project.dtypes 
print(df_project['Status'].unique())
df_project['balance'] = [1 if Status=='STOLEN' else 0 for Status in df_project.Status]
df_project['balance'].value_counts()

##balance the original dataset by Down-sample Majority Class
from sklearn.utils import resample
# Separate majority and minority classes
df_majority = df_project[df_project.balance==0]
df_minority = df_project[df_project.balance==1]
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=1000,    #set a reasonable number
                                 random_state=123) # reproducible results
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=True,    # sample without replacement
                                 n_samples=1000,     # to match minority class
                                 random_state=123) # reproducible results
df_balanced = pd.concat([df_majority_downsampled, df_minority_upsampled])
df_balanced = df_balanced.drop('balance', axis=1)
df_balanced = df_balanced.drop('Location_Type', axis=1)
df_balanced.shape
df_balanced.dtypes 
df_balanced['Premise_Type'].unique()
df_balanced['Bike_Type'].unique()
df_balanced['Cost_of_Bike'].unique()
df_balanced['Status'].unique()
df_balanced['Neighbourhood'].unique()

# check the null values
print(df_balanced.isnull().sum())
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

# Separate input features (X) and target variable (y)
x = scaled_df
y = df_balanced['Status']
# Split the data into train test
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)

#build the tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
for i in range(5,20):
    dt1_project = DecisionTreeClassifier(criterion='entropy',max_depth=i, min_samples_split=20, random_state=123)
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

import joblib 
joblib.dump(dt1_project, '/Users/xiaojing/5thSemester/Datawarehouse/project/projectmodel_lr.pkl')
print("Model dumped!")

model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, '/Users/xiaojing/5thSemester/Datawarehouse/project/projectmodel_columns.pkl')
print("Models columns dumped!")








