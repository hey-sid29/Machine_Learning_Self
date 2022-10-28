"""Calculation of Euclidian Distance to build a KNN algo from scratch"""


from math import sqrt


q1=[1, 3]

p1=[2, 5]

## Double dimension data so i= 1 to 2   (0,1)-> dims

euclidean_distance= sqrt( (q1[0] - p1[0])**2 + (q1[1] - p1[1])**2 )

print(euclidean_distance)