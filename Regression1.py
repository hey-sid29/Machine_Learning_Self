"""The idea of regression is to take in continous data and draw a best fit line; 
which basically boils down into modelling your data"""


import pandas as pd
import quandl as ql
data=ql.get('WIKI/GOOGL')
#pd.set_option('display.max_columns', 12)
#print(data.head())
data1=data[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#Your first job for building a regressive model is to look for the datasets which have
#relevant data or relation between datas; as in this case we dont need all the columns of the stock prices
#we just need the relevent columns like Adjusted columns

#print(data1.head())
data1['HL_PCT']=( (data1['Adj. High']-data1['Adj. Low'])/data1['Adj. Low'] )* 100.0
#HL_PCT= High to Low Percent change= (New Price(adj. high)- Old Price(adj. Low))/Old Price(adj. Low)*100.0

data1['PCT_change']=( (data1['Adj. Close']-data1['Adj. Open'])/data1['Adj. Open'] )* 100.0
#PCT_Change= Total Percent Change= 
# [(Closing Price of the day{Adj. CLose}-Opening Price for the day{Adj. Open})/Opening Price{Adj. Open}]*100.0
data1=data1[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(data1.head())