import pickle
import math, quandl, datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing
import matplotlib.pyplot as plt
from matplotlib import style
#Importing all necessary libraries
style.use('ggplot')
#Data Source for ML: Quandl.com; stocks of google

df = quandl.get('WIKI/GOOGL')
#print(df.head(5))

df= df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']= ( (df['Adj. High']-df['Adj. Low'])/df['Adj. Low']) * 100.0
df['PCT_Change']= ((df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']) * 100.0

df= df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
df.fillna(value=-9999, inplace=True)
forecast_col= 'Adj. Close'
forecast_out= int(math.ceil(0.01* len(df)))

df['Label']= df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
#print(df.head())

#In training and testing: 'X' denotes the Features and, 'y' denotes the labels

X=np.array( df.drop(['Label'], axis=1) )  #X would consist of all the meaningful data thats necessary for prediction

X= preprocessing.scale(X)   #Scaling data is helpful for short amount of datasets

X_lately=X[-forecast_out:]
#X=X[:-forecast_out]

df.dropna(inplace=True)



y= np.array(df['Label'].values)
#y= df['Label']



X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2)  #Preparing our train and test data

#We are not keeping the test size for train data as -forecast_out because we already shifted the data acc to it
#Same for test data as the main df was shifted by the size of -forecast_out

clf= LinearRegression(n_jobs=-1) #Preparing our 1st Regression model
#For note purposes: LinearRegression can be threaded simply: 
#n_jobs parameter will set the threading values
#setting n_jobs=-1 implies that we are runnig all jobs that can be done by our CPU 
#threading= is the process of completing as many jobs possible by CPU's cores
#Threading is simple for LR and can be run on massive datasets but its not simple as for Support Vector Machines




clf.fit(X_train, y_train) #training the classifier 
#At this point we can pickle our own Classifier to save the tedious time to train our classifier


#with open("LinearRegression1.pkl", "wb") as file:
    #pickle.dump(clf, file)

    
#now the clf is saved(pickled) and can be used anytime

#Retrieving the pickled clf

inp= open("LinearRegression1.pkl", "rb")
clf= pickle.load(inp)




accuracy = clf.score(X_test, y_test) #checking accuracy

forecast_set= clf.predict(X_lately)

#print(forecast_set, accuracy, forecast_col)


#Hard coding all the days for prediction values

df['Forecast']= np.nan
last_date= df.iloc[-1].name
last_unix= last_date.timestamp()
one_day=86400   #Number of hrs in one day
next_unix= last_unix+one_day

#Populating the dataframe with values from forcast_set

for i in forecast_set:
    next_date= datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)] + [i]
print(df.head())
    
    
## Explanation for the for loop:
"""next_date will store the dates generated from next_unix,
    now next_unix is updated by every one day value
    now the next_date valued column is firstly filled with np.nan values which gets replaced by the [i] which 
    is each element in the forecast_set"""

#Plotting A Plot for the predicted prices

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc='lower right')
plt.xlabel("Dates")
plt.ylabel("Prices")
plt.savefig("Stock_Predict.png")
plt.show()
