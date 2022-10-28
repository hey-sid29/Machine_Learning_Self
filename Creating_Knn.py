"""Creating Knn algo from scratch"""

from collections import Counter
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings #To warn user if they use a useless number for K
from matplotlib import style


style.use('fivethirtyeight')

dataset = {'k': [[1,2], [2,3], [3,1]], 'r': [[5,6], [6,7], [7,5]]}
#2D dataset = (x,y) 
#dataset contains the features for our knn algo

new_data= [4,5]

#label or prediction data for the model

"""[[plt.scatter(ii[0], ii[1], s=100, c=i) for ii in dataset[i]] for i in dataset]"""   


#A nested for Loop; to iterate over every element in the dataset 
#ii[0] represents the X-value/X-coordinate
#ii[1] represents Y-value/Y-coordinate


"""plt.scatter(new_data[0], new_data[1], color='yellow', s=110) """

#to generate a general visual image of how Knn works

"""plt.show()"""

#Check KNN1.png




"""Creating KNN function and model: """

def k_nearest_neighbors(dataset, predict, k=3):
    
    #Breakdown of the function:

    #1. dataset= this param is the training data ie. the data on which our own model would be trained
    #(SET OF FEATURES)  *contd. 1

    #2.predict = this is the testing data; the data our model will predict
    #(SET OF LABELS) *contd. 2
    
    #3.k= contains the number of points to compare the test data to.
    # k!=2n (n=1,2,3.....)  xxxx
    # k==2n-1 (n=1,2,3....) oooo




    if len(dataset)>=k:

        warnings.warn(f"K = {k} is set to be less than the number of all the voting groups available!")
    
    dist = []
    for group in dataset:
        for feature in dataset[group]:

            """distances = np.sqrt(np.sum((np.array(feature) - np.array(predict))**2))"""

            #Formula for Euclidean distances for Dynamic Dimensional Array^
            #Formula for Euclidean distances for 2D array/dataset in "euclidian_dist.py"

            #note: as that formula was hard-coded for 2d arrays or dataset it will be very complex to do that for 
            #     dynamic dimensional arrays

            # THE ABOVE FORMULA IS NOT VERY EFFECIENT
            ec = np.linalg.norm(np.array(feature) - np.array(predict))
            dist.append([ec, group])

    #Now we have to check through the distances list and grab the lowest ones:

    votes= [i[1] for i in sorted(dist)[:k]]

    #breakdown:
    #i[1] basically relates to the 'group' passed
    #sorting helps us to have the least distances first 
    #k helps us reduce the number of points to compare for ex: k=3; the first 3 SORTED distances

    vote_result= Counter(votes).most_common(1)[0][0]   
    #[0][0] without the 0,0 the returning value would be [(group, votes)]
    #the 0,0 returns the "group" 

    
    return vote_result

res = k_nearest_neighbors(dataset, predict=new_data, k=3)
print(res)

[[plt.scatter(ii[0], ii[1], s=100, c=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_data[0], new_data[1], color=res, s=110)
plt.show()

#See KNN2 for ref.


