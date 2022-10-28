"""Creating a Simple Regressive Algorithm to plot a best fit straight line"""
"""Formula for Mean= [mean(x).mean(y) - mean(xy)]/[mean(x)^2 - mean(x^2)]"""
"""formula for y-intercept: b= mean(y) - m*mean(x)"""

# Trying to plot a simple scatter plot from simple data

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype= np.float64)
ys = np.array([9, 10, 11, 7, 9, 8], dtype=np.float64)

# Now we make a function that returnes us the best fit slope


def best_fit_slope_and_intercept(xs, ys):

    #1. calculating slope(m)

    m = ((mean(xs)*mean(ys)) - mean(xs*ys))/((mean(xs)*mean(xs)) - mean(xs**2))


    #2. Calculating y-intercepts(b)

    b= mean(ys)-m*mean(xs)


    return m,b


m,b = best_fit_slope_and_intercept(xs, ys)
print(m,b)

#Now how do we plot a best-fit line?? We know the line equation which is: mx+b so we need a list/array of values for xs

regression_line= np.array([((m*x)+b) for x in xs], dtype=np.float64)

#Plotting the regression line on the scatter plot

plt.scatter(xs, ys)  #the scatter plot for raw data

plt.plot(xs, regression_line) #plotting the regressive line against xs


plt.show()
