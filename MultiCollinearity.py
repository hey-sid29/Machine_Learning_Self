"""MultiCollinearity In Linear Regression"""
import pandas as pd
import statsmodels.api as sm
# import matplotlib.pyplot as plt

"""

df = pd.read_csv('archive/Advertising.csv', index_col=0)
# print(df)
X = df.iloc[:, :-1]
y = df.iloc[:, 3]

# This is the case of multiple regression;
# Independent Features: TV, Radio, Newspaper
# Dependent Feature(Target): Sales
# Regression eqn: y = m1*(TV) + m2*(Radio) + m3*(Newspaper) + C

# Training an OLS model[Ordinary Least Squares]; In OLS, value for C = 1

X = sm.add_constant(X)
ols_model = sm.OLS(y, X).fit()
# print(ols_model.summary())"""

"""                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.897
Model:                            OLS   Adj. R-squared:                  0.896
Method:                 Least Squares   F-statistic:                     570.3
Date:                Tue, 25 Oct 2022   Prob (F-statistic):           1.58e-96
Time:                        19:45:05   Log-Likelihood:                -386.18
No. Observations:                 200   AIC:                             780.4
Df Residuals:                     196   BIC:                             793.6
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.9389      0.312      9.422      0.000       2.324       3.554
TV             0.0458      0.001     32.809      0.000       0.043       0.049
Radio          0.1885      0.009     21.893      0.000       0.172       0.206
Newspaper     -0.0010      0.006     -0.177      0.860      -0.013       0.011
==============================================================================
Omnibus:                       60.414   Durbin-Watson:                   2.084
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              151.241
Skew:                          -1.327   Prob(JB):                     1.44e-33
Kurtosis:                       6.332   Cond. No.                         454.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."""

# We see that all the independent features are not really correlated

# print(X.iloc[:, 1:].corr())

# Correlation Matrix:
"""                 TV     Radio  Newspaper
TV         1.000000  0.054809   0.056648
Radio      0.054809  1.000000   0.354104
Newspaper  0.056648  0.354104   1.000000
"""
"""------------------------------------------------------------------------------------------------------------------"""

# hence we can conclude this dataset has less correlation == no multi collinearity


df_salary = pd.read_csv("Salary_Data.csv")
# print(df_salary)

X = df_salary[['YearsExperience', 'Age']]
y = df_salary['Salary']


X = sm.add_constant(X)
ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

"""                           OLS Regression Results                            
==============================================================================
Dep. Variable:                 Salary   R-squared:                       0.960
Model:                            OLS   Adj. R-squared:                  0.957
Method:                 Least Squares   F-statistic:                     323.9
Date:                Tue, 25 Oct 2022   Prob (F-statistic):           1.35e-19
Time:                        20:11:36   Log-Likelihood:                -300.35
No. Observations:                  30   AIC:                             606.7
Df Residuals:                      27   BIC:                             610.9
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
const           -6661.9872   2.28e+04     -0.292      0.773   -5.35e+04    4.02e+04
YearsExperience  6153.3533   2337.092      2.633      0.014    1358.037    1.09e+04
Age              1836.0136   1285.034      1.429      0.165    -800.659    4472.686
==============================================================================
Omnibus:                        2.695   Durbin-Watson:                   1.711
Prob(Omnibus):                  0.260   Jarque-Bera (JB):                1.975
Skew:                           0.456   Prob(JB):                        0.372
Kurtosis:                       2.135   Cond. No.                         626.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
print(X.iloc[:, 1:].corr())

"""               YearsExperience       Age
YearsExperience         1.000000  0.987258
Age                     0.987258  1.000000"""

# 1) we see age and Years Of Experience are correlated upto 98%, which is very high;
# 2) So we can drop Age as The YearsOfExperience covers 98% properties of Age.
