"""Multiple Linear Regression"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("50_Startups.csv")
# print(df.head(10))

"""DataFrame preview
    R&D Spend  Administration  Marketing Spend       State     Profit
0  165349.20       136897.80        471784.10    New York  192261.83
1  162597.70       151377.59        443898.53  California  191792.06
2  153441.51       101145.55        407934.54     Florida  191050.39
3  144372.41       118671.85        383199.62    New York  182901.99
4  142107.34        91391.77        366168.42     Florida  166187.94
5  131876.90        99814.71        362861.36    New York  156991.12
6  134615.46       147198.87        127716.82  California  156122.51
7  130298.13       145530.06        323876.68     Florida  155752.60
8  120542.52       148718.95        311613.29    New York  152211.77
9  123334.88       108679.17        304981.62  California  149759.96"""

# Separating the Dependent and Independent features

# independent features:  R&D spend, Administration, Marketing, State
# dependent features: Profit

X = df.iloc[:, :-1]  # all columns except the Profit column
y = df.iloc[:, 4]  # Profit col is at index == 4

# Encoding all the categorical values in State column:

states = pd.get_dummies(df['State'], drop_first=True)
# This statement will create dummy columns of the names in State;
# Then drop_first will drop the first column to avoid Dummy Variable Trap condition.

X = X.drop('State', axis=1)
# Dropping the state column

X = pd.concat([X, states], axis=1)
# Joining the encoded columns with cleaned dataframe

# print(X.head(10))
"""New formatted Dataframe with encoded columns:
    R&D Spend  Administration  Marketing Spend  Florida  New York
0  165349.20       136897.80        471784.10        0         1
1  162597.70       151377.59        443898.53        0         0
2  153441.51       101145.55        407934.54        1         0
3  144372.41       118671.85        383199.62        0         1
4  142107.34        91391.77        366168.42        1         0
5  131876.90        99814.71        362861.36        0         1
6  134615.46       147198.87        127716.82        0         0
7  130298.13       145530.06        323876.68        1         0
8  120542.52       148718.95        311613.29        0         1
9  123334.88       108679.17        304981.62        0         0
"""

# Splitting dataset into training and test set in the ratio of 80%-20%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Training a Linear Regression model
lnr_reg = LinearRegression()
lnr_reg.fit(X_train, y_train)

# Calculating predictions on X_test

y_pred = lnr_reg.predict(X_test)

# Now calculating the R^2_score on y_test vs. y_pred
score = r2_score(y_test, y_pred)
print(score)

"""Accuracy of model: 0.9297662471341652 
or, 92.97%"""


