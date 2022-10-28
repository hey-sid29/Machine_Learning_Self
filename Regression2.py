"""Regression: Features and labels
    - Features: Are the input variables. X in regression
    - Labels: Are the values which we are Predicting. In Regression the Y-variable"""

import math
import pandas as pd
import quandl as ql
data=ql.get('WIKI/GOOGL')
data1=data[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
data1['HL_PCT']=( (data1['Adj. High']-data1['Adj. Low'])/data1['Adj. Low'] )* 100.0
data1['PCT_change']=( (data1['Adj. Close']-data1['Adj. Open'])/data1['Adj. Open'] )* 100.0
data1=data1[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
data1.fillna(-99999, inplace=True)
forecast_col='Adj. Close' #Just a variable to help out the forecast(prediction)
forecast_out=math.ceil(0.01*len(data1))
data1['New Label']=data1[forecast_col].shift(-forecast_out)
data1.dropna(inplace=True)
print(data1.head(5), '\n\n', data1.tail(5))

"""Basic Regressive Model^^"""