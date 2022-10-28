"""Calculating the best fit line, Slope, and y-intercepts on a sample input data"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


# Creating a Sample Datasets:

def create_datasets(value, variance, step_up=2, correlation=False):


    #value to plot HOW MANY data points
    #variance to variate the values
    #step_up is by how much the y should step up
    #correlation between the data points can be +ve, -ve, None
    
    val=1
    ys=[]
    for i in range(value):
        y= val+random.randrange(-variance, +variance)
        ys.append(y)
        ## The values of ys would not variate as desired so we have to create the variation which favours our 
        #motive

        if correlation=='pos':
            val+= step_up
        elif correlation=='neg':
            val-=1
    
    xs=[i for i in range(len(ys))]
    return (np.array(xs, dtype=np.float64)), (np.array(ys, dtype=np.float64))



def best_fit_slope_and_intercept(xs, ys):
    
    #1. calculating slope(m)

    m = ((mean(xs)*mean(ys)) - mean(xs*ys))/((mean(xs)*mean(xs)) - mean(xs**2))


    #2. Calculating y-intercepts(b)

    b= mean(ys)-m*mean(xs)


    return m,b

def squared_error(orig_ys, line_ys):
    return sum((line_ys-orig_ys)**2)

    #Since e^2 = (y on line - y originals)^2


def calc_r_square(orig_ys, line_ys):
    
    #We need a mean line of all the original y
    ys_mean_line= [mean(orig_ys) for y in orig_ys]

    squared_error_regr= squared_error(orig_ys, line_ys) 

    #^^ above line will calculated the square error for the regressive line

    squared_error_mean= squared_error(orig_ys, ys_mean_line)

    return 1- (squared_error_regr / squared_error_mean)   # <- is the  value of r-square or coeffecient of determination
    

xs,ys= create_datasets(value=50, variance= 40, step_up=4, correlation='pos')

m,b = best_fit_slope_and_intercept(xs, ys)
print(m,b)

#Now how do we plot a best-fit line?? We know the line equation which is: mx+b so we need a list/array of values for xs

regression_line= np.array([((m*x)+b) for x in xs], dtype=np.float64)

r_square= calc_r_square(ys, regression_line)
print(r_square)



#Plotting the regression line on the scatter plot

plt.scatter(xs, ys)  #the scatter plot for raw data

plt.plot(xs, regression_line, label= 'Best Fit Line') #plotting the regressive line against xs
plt.legend(loc= 'lower left')

### Plotting the scatter plot and regressive line on sample data and calculating the r_square based on it

plt.show()
