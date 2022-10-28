"""Classifier: K-Nearest Neighbors(KNN)"""

"""Cancer Prediction model"""


from sklearn import preprocessing, model_selection, neighbors
import numpy as np
import pandas as pd


df = pd.read_csv('breast-cancer-wisconsin.data') #, index_col=0)
# reading in the csv-file

df.replace('?', -99999, inplace=True) 
#Replacing the missing data marked by '?' by -99999 so it makes a greater outlier and the machine practically ignore
#this numeric data

#Now after the cleaning, our main job is to find the relevant and irrelevant data columns
#Since in a breast cancer predictive model using KNN we dont need ID's so we will drop it
#axis=1 because we need to drop columns 


df.drop(['id'], axis=1, inplace=True)   # pr we could set index_col=0 in line 9 

#Creating our features and labels:

X= np.array(df.drop(['class'], 1))  #dropping the class column since it is the label

y= np.array(df['class']) #creating label array


#Training testing the data

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train) 

accuracy = clf.score(X_test, y_test)
print(accuracy)

#Output: 0.9785714285714285 => 97.8%
#Output 2: 0.9928571428571429 => 99.2%

example_data= np.array([1,6,7,9,8,2,4,5,3])
example_data= example_data.reshape(1, -1)

predict = clf.predict(example_data)
print(predict)

#Output: predict= [4]; which is the category of malingnant

print("Sample 2 data")

example_data2= np.array([2,4,1,1,1,3,2,5,3])
example_data2= example_data2.reshape(1, -1)


predict2= clf.predict(example_data2)
print(predict2)


#Sample 2 data Output:
#[2] => benign


### Sample Data 3: 

print('Sample Data 3: Multiple Patients')

ex_data3= np.array([[6,9,7,8,4,2,3,5,2], [8,9,7,4,6,2,3,5,8]])
ex_data3= ex_data3.reshape(len(ex_data3), -1)
#the method of reshaping is best here as this method -> array.reshape(len(array), -1) <- as for
#future purposes we dont know how many patients sample we are checking at once, so len of array will keep on changing
#if we increase patients,
# and -1 gives an unknown dimension which numpy and scikit learn calculates for acc to its need and saves us the time
#to hard code




predict3= clf.predict(ex_data3)
print(predict3)

"""Sample Data 3: Multiple Patients/Output:-

    [4 4]=> malingnant - malingnant"""


