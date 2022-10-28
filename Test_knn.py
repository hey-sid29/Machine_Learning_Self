"""Applying Our Build of KNN model onto real world data: The breast cancer dataset from UCI dataset repo"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
#dont forget this
import pandas as pd
import random
style.use('fivethirtyeight')

def k_nearest_neighbors(data, predict, k=3):
        
        
    #Breakdown of the function:

    #1. dataset= this param is the training data ie. the data on which our own model would be trained
    #(SET OF FEATURES)  *contd. 1

    #2.predict = this is the testing data; the data our model will predict
    #(SET OF LABELS) *contd. 2
    
    #3.k= contains the number of points to compare the test data to.
    # k!=2n (n=1,2,3.....)  xxxx
    # k==2n-1 (n=1,2,3....) oooo


    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:

            """distances = np.sqrt(np.sum((np.array(feature) - np.array(predict))**2))"""

            #Formula for Euclidean distances for Dynamic Dimensional Array^
            #Formula for Euclidean distances for 2D array/dataset in "euclidian_dist.py"

            #note: as that formula was hard-coded for 2d arrays or dataset it will be very complex to do that for 
            #     dynamic dimensional arrays

            # THE ABOVE FORMULA IS NOT VERY EFFECIENT

            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    
    #Now we have to check through the distances list and grab the lowest ones:

    votes = [i[1] for i in sorted(distances)[:k]]
    #breakdown:
    #i[1] basically relates to the 'group' passed
    #sorting helps us to have the least distances first 
    #k helps us reduce the number of points to compare for ex: k=3; the first 3 SORTED distances

    vote_result = Counter(votes).most_common(1)[0][0]
    #[0][0] without the 0,0 the returning value would be [(group, votes)]
    #the 0,0 returns the "group" 


    confidence = Counter(votes).most_common(1)[0][1] / k  #[0][1] will return the 'votes'
    return vote_result, confidence

accuracies = [] 


for i in range(20):
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?',-99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    #Converting the dataset to float-type as py treats some of the cols as 'str'- which might reduce the 
    #overall effeciency of model

    random.shuffle(full_data)  #Shuffling the data

    test_size = 0.5
    train_set = {2:[], 4:[]}
    #train_set will be an empty dict 2,4 will be the classes benign and malignant respectively

    test_set = {2:[], 4:[]}
    #the set which will carry the predicted data

    train_data = full_data[:-int(test_size*len(full_data))]
    #-int(test_size * len(full_data) creates an index to where the train_data is parsed
    #in here: its from the start to -int(test_size * len(full_data)

    test_data = full_data[-int(test_size*len(full_data)):]
    #index: from -int(test_size * len(val_data) to the last column

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
        ## To populate our dict with values we have grab the class col.
        ## i[-1] is for that: either 2 for benign tumors will get populated
        ##                       or 4 for malignant tumors will get populated



    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0        #correct predictions
    total = 0          #total predictions'
    #print(len(test_set))
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)

            #k_nearest_neighbors(dataset, predict, k=3):
            #dataset = train_data; this is the training sample
            #predict = data; is the data(predictable) in the test set
            #k=5; is the default scikit's K values

            if group == vote:
                correct += 1
            else: 
                pass    #denotes the wrong votes
            total += 1
    #print('Accuracy:', correct/total)
    accuracies.append(correct/total)


print(sum(accuracies)/len(accuracies))
