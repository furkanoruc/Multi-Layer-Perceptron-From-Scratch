# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:09:39 2020

@author: User
"""

# pip install python-mnist will install the required package

import pandas as pd
import matplotlib.pyplot as plt
from mnist import MNIST
import numpy as np
import random
import math

np.random.seed(60) # reproducability
mndata = MNIST('./mnist')

#Added in order to read the file successfully, to decompress file.
mndata.gz = True

# read training images and corresponding labels
tr_images, tr_labels = mndata.load_training()
# read test images and corresponding labels
tt_images, tt_labels = mndata.load_testing()

# convert lists into numpy format and apply normalization
tr_images = np.array(tr_images) / 255. # shape (60000, 784)
tr_labels = np.array(tr_labels)         # shape (60000,)
tt_images = np.array(tt_images) / 255. # shape (10000, 784)
tt_labels = np.array(tt_labels)         # shape (10000,)

""""Consider Randomness for xt & rt if needed, later on
#randomness of t
np.random.seed(60) # reproducability
r = list(range(10000))
random.shuffle(r)
print(r)"""

#Desired Training

f = 0
t = 0
holder = 0

desired_training = pd.DataFrame(index=range(60000),columns=range(10))
desired_training = desired_training.values

for f in range (0,60000):
    holder = tr_labels[f]
    for t in range (10):
        if t != holder:
            desired_training[f][t] = 0
        elif t == holder:
            desired_training[f][holder] = 1



ts = 0
i = 0
t = 0
p = 0
j = 0
g = 0
k = 0
v = 0
r = 0
loss = 0
holder_list = []
error = []
accuracy = []
w = []
o = []
o = pd.DataFrame(index=range(1),columns=range(10))
o = o.values
o_test = []
o_test = pd.DataFrame(index=range(1),columns=range(10))
o_test = o_test.values
o_exp_sum = 0
y = []
y = pd.DataFrame(index=range(1),columns=range(10))
y=y.values
y_function = []
y_function = pd.DataFrame(index=range(1),columns=range(10))
y_function = y_function.values
y_test = []
y_test = pd.DataFrame(index=range(1),columns=range(10))
y_test = y_test.values
y_holder = []
y_holder = pd.DataFrame(index=range(60000),columns=range(10)) #change for training instance
y_holder = y_holder.values
y_holder_test = []
y_holder_test = pd.DataFrame(index=range(10000),columns=range(10)) #change for test instance
y_holder_test = y_holder_test.values
epoch = []
epoch = pd.DataFrame(index=range(60000),columns=range(1)) #change for training instance
epoch = epoch.values
epoch_test = []
epoch_test = pd.DataFrame(index=range(10000),columns=range(1)) #change for test instance
epoch_test = epoch_test.values
soft = []
soft = pd.DataFrame(index=range(1),columns=range(10))
soft=soft.values
w = pd.DataFrame(index=range(10),columns=range(784))
w=w.values

#Softmax
def softmax(t, r=0):
    for i in range (0,10):
        r = r + (math.exp(t[0][i]))
    for i in range (0,10):
        y_function[0][i] = math.exp(t[0][i]) / r
    return y_function


l=(softmax(o))
l.sum()

for i in range (0,10):
    for j in range (0,784):
        w[i][j] = random.uniform(-0.01, 0.01)

bias = np.random.uniform(-0.01, 0.01, 10)


for q in range (0,50): #Number of Epochs?
    epoch[q,0] = 0
    epoch_test[q,0] = 0 #test
    for t in range (0,60000): #Instances for Training Dataset: Change Based on number of instances.
        o[0,:] = 0
        o_test[0,:] = 0 #test
        o = o + (np.dot(w, tr_images[t])) + bias
        if t < 10000:
            o_test = o_test + (np.dot(w, tt_images[t])) + bias #test
        y = (softmax(o))
        y_holder[t,:] = y[0,:]
        if t < 10000:
            y_test = (softmax(o_test)) #test
            y_holder_test[t,:] = y_test[0,:] #test
        for i in range (0,10):
            loss = desired_training[t,i] - y[0,i]
            w[i] = w[i] + 0.0001*(loss)*tr_images[t]
        if tr_labels[t] == np.argmax(y_holder[t]):
            epoch[q,0] = epoch[q,0] + 1
        if t < 10000 and tt_labels[t] == np.argmax(y_holder_test[t]): #test
            epoch_test[q,0] = epoch_test[q,0] + 1    
    q = q + 1 
    

plt.plot(epoch/60000)
plt.title('accuracy per epoch - training - %')
plt.show()

plt.plot(epoch_test/10000)
plt.title('accuracy per epoch - test - %')
plt.show()

#confusion matrix

con_y_holder= y_holder.copy()
con_y_holder_test = y_holder_test.copy()

confusion_train = []
confusion_train = pd.DataFrame(index=range(10),columns=range(10))
confusion_train = confusion_train.values
confusion_test = []
confusion_test = pd.DataFrame(index=range(10),columns=range(10))
confusion_test = confusion_test.values
conf = 0
conf_label = 0
conf_assigned = 0

#training confusion matrix prep

for i in range (0,10):
    for j in range (0,10):
        confusion_train[i][j] = 0


#training confusion matrix
for i in range (0,60000):
    if np.argmax(y_holder[i]) == tr_labels[i]:
        conf_label = tr_labels[i]
        confusion_train[conf_label,conf_label] = confusion_train[conf_label,conf_label] + 1
    elif np.argmax(y_holder[i]) != tr_labels[i]:
        conf_assigned = np.argmax(y_holder[i])
        conf_label = tr_labels[i]
        confusion_train[conf_label,conf_assigned] = confusion_train[conf_label,conf_assigned] + 1

#test confusion matrix
conf_label = 0
conf_assigned = 0

for i in range (0,10):
    for j in range (0,10):
        confusion_test[i][j] = 0

for i in range (0,10000):
    if np.argmax(y_holder_test[i]) == tt_labels[i]:
        conf_label = tt_labels[i]
        confusion_test[conf_label,conf_label] = confusion_test[conf_label,conf_label] + 1
    elif np.argmax(y_holder_test[i]) != tt_labels[i]:
        conf_assigned = np.argmax(y_holder_test[i])
        conf_label = tt_labels[i]
        confusion_test[conf_label,conf_assigned] = confusion_test[conf_label,conf_assigned] + 1


        
df_confusion_test = pd.DataFrame(confusion_test)
df_confusion_train = pd.DataFrame(confusion_train)



w_transpose = w.transpose()
#Visualization Test

#from matplotlib import pyplot
import numpy as np
#import csv
import matplotlib.pyplot as plt
k = 0
for k in range (0,10):
    pixels_mean = w_transpose[0:]
    pixels_mean = np.array(pixels_mean, dtype='uint8')
    deneme = pixels_mean[:,k].reshape(28, 28) #CHANGE k based on numbers
    plt.title('Weight Visualization: Label ' + str(k))
    plt.imshow(deneme, cmap='gray')
    plt.show()

deneme = []

