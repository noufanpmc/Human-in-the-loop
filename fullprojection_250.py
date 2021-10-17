# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:57:27 2021

@author: Superviseur
"""


from lms import *
from saw import *
from stm2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import cv2 as cv
import time
import os
from random import *
import math

from sklearn import metrics
from sklearn.metrics import accuracy_score
from imutils import face_utils
import imutils
import dlib
import keypoints_250 as kp250 #imporing the the python script


 
def multiplyLists(l1, l2) :
    """
        Multiply two lists one to one
    """
        
    l3 = [0.0 for i in range(len(l1))]
    for i in range(len(l1)) :   #for each neurons of the network
        l3[i] = l1[i]*l2[i]
    return l3   


def createInputVector(neurons, size=10) :
    """
        Function used in order to transform neurons' output in a vector
        The activity of the neurons must be previously calculated (see calculate_activity)
    
        Parameters : 
        neurons -- the list of neurons
        size -- the size of desired vector (10 by default)
    """
    
    inputVector = []
    for i in range(size) :      #for each part of the vector
        if(i<len(neurons)) :    #if there are still neurons, put the neuron's activity
            inputVector.append(neurons[i].activity)
        else :                  #else put a null value
           inputVector.append(0) 
    return inputVector
    

def neuronsToArray(neurons, rows, cols) :
    """
        Function used in order to transform neurons' output in a matrix
        The weighting of the neurons must be previously calculated (see calculate_weightings)
    
        Parameters : 
        neurons -- the matrix of neurons
        rows -- the number of neurons of the network in rows
        cols -- the number of neurons of the network in columns
    """
    
    array = np.zeros((rows,cols))
    for i in range(rows) :        #for each neurons of the network in rows
        for j in range(cols) :    #for each neurons of the network in columns
            array[i][j] = neurons[i][j].weighting
        
    return array

# to create abbrevatipn of expressions
def make_abv(y_list):
    new_y_list = []
    for i in y_list:
        if i == "colere":
            j = i.replace("colere", 'C')
            new_y_list.append(j)
        elif i == "joie":
            j = i.replace("joie", 'J')
            new_y_list.append(j)
        elif i == "neutre":
            j = i.replace("neutre", 'N')
            new_y_list.append(j)
        elif i == "surprise":
            j = i.replace("surprise", 'S')
            new_y_list.append(j)
        elif i == "tristesse":
            j = i.replace("tristesse", 'T')
            new_y_list.append(j)
    return new_y_list

def shuffle_list(*ls):
  l =list(zip(*ls))

  shuffle(l)
  return zip(*l)


def unlist_list(a, b): #to unlist the list of lists
    flat_list_a = []
    for i in a:
        for j in i:
            flat_list_a.append(j)
    flat_list_b = [j for i in b for j in i]
    a1,b1 = shuffle_list(flat_list_a,flat_list_b)
    a1 = list(a1)
    b1 = list(b1)
    
    m=0
    new_a1=[]
    while m<len(a1):
      new_a1.append(a1[m:m+25])
      m+=25
    new_a1 = list(new_a1)   
    new_a1 = [list(ele) for ele in new_a1] 
    
    n=0
    new_b1=[]
    while n<len(a1):
      new_b1.append(b1[n:n+25])
      n+=25
    new_b1 = list(new_b1)   
    new_b1 = [list(ele) for ele in new_b1]
    
    return new_a1, new_b1
                

'''
MAIN 
'''
seed(42)
expressions = ['colere', 'joie', 'neutre', 'surprise', 'tristesse']

list_jpg = [[f for f in os.listdir() if f[:len(exp)]==exp] for exp in expressions]#list of different emotions
num = [[int(f[len(expressions[i])+1:-4]) for f in list_jpg[i]] for i in range(len(list_jpg))]#list of their numbers
list_jpg = [[a for b, a in sorted(zip(num[i],list_jpg[i]))] for i in range(len(list_jpg))] #sort by increase
kp_all = kp250.kp_all
# list_jpg = [sample(i, len(i)) for i in list_jpg] 
# shuffle(list_jpg)

#unlisting all the list of lists for shuffling 
#unlisting the images
list_1 = list_jpg[0]
list_2 = list_jpg[1]
list_3 = list_jpg[2]
list_4 = list_jpg[3]
list_5 = list_jpg[4]

#unlisting the keypoints
kp_1 = kp_all[0]
kp_2 = kp_all[1]
kp_3 = kp_all[2]
kp_4 = kp_all[3]
kp_5 = kp_all[4]

# Shuffling corresponding images and keypoints
list_1_shf, kp_1_shf = shuffle_list(list_1, kp_1)
list_1_shf = list(list_1_shf)
kp_1_shf = list(kp_1_shf)

list_2_shf, kp_2_shf = shuffle_list(list_2, kp_2)
list_2_shf = list(list_2_shf)
kp_2_shf = list(kp_2_shf)

list_3_shf, kp_3_shf = shuffle_list(list_3, kp_3)
list_3_shf = list(list_3_shf)
kp_3_shf = list(kp_3_shf)

list_4_shf, kp_4_shf = shuffle_list(list_4, kp_4)
list_4_shf = list(list_4_shf)
kp_4_shf = list(kp_4_shf)

list_5_shf, kp_5_shf = shuffle_list(list_5, kp_5)
list_5_shf = list(list_5_shf)
kp_5_shf = list(kp_5_shf)


# adding to a new list of lists after shuffling
list_jpg_shuffled = []
list_jpg_shuffled.append(list_1_shf)
list_jpg_shuffled.append(list_2_shf)
list_jpg_shuffled.append(list_3_shf)
list_jpg_shuffled.append(list_4_shf)
list_jpg_shuffled.append(list_5_shf)


kp_all_shuffled = []
kp_all_shuffled.append(kp_1_shf)
kp_all_shuffled.append(kp_2_shf)
kp_all_shuffled.append(kp_3_shf)
kp_all_shuffled.append(kp_4_shf)
kp_all_shuffled.append(kp_5_shf)

# list_jpg_new, kp_all_new = unlist_list(list_jpg, kp_all)

#creating the learning and testing base for images
learning_base = [l[:len(l)-10] for l in list_jpg_shuffled]
test_base = [l[len(l)-10:] for l in list_jpg_shuffled]

#creating the learning and testing base for keypoints
kp_all_learn = [l[:len(l)-10] for l in kp_all_shuffled]
kp_all_test = [l[len(l)-10:] for l in kp_all_shuffled]



nbFramesIntegrated = 1
# norm = 1.0/nbFramesIntegrated

nbFeatures = 128 #128                             #The number of descriptor used in a point of the image
nbPoints = 8  #10                               #The number of point used in an image

sawSizeImage = 500 #300                       #The maximum size of a SAW for the image

vigilanceSawImage = 0.98#0.96

"""q
file_saw_image = open('saw_image.obj', 'r') 
sawImage = pickle.load(file_saw_image)                  #Creation of a SAW for the visual modality

file_lms_image = open('lms_image.obj', 'r') 
lmsImage = pickle.load(file_lms_image)                  #Creation of a LMS for the visual modality

stmImage = STM(5, 1.0/nbPoints)                         #Creation of a STM(number of neurons) for the visual modality

"""

sawImage = SAW(nbFeatures, vigilanceSawImage, sawSizeImage, 0.01)#0.01#Creation of a SAW(dimension, vigilance threshold, neurons maximum, learning rate)
lmsImage = LMS(sawSizeImage, 5, 0.1, rangeMin=-0.3, mangeMax=0.3)#0.1    #Creation of a LMS(dimension, number of neurons, learning rate)
stmImage = STM(5, 1.0/nbPoints)                         #Creation of a STM(number of neurons) for the visual modality
stmSliding = STM(5, 1.0, 0.8) 

inputImage = []                        #The descriptors input(matrix)
desiredOutput = [0,0,0,0,0]

#Capture 
t = 0
#tps1 = time.clock()
tps1 = time.perf_counter()
tps2 = tps1
duree_app = 150
index,j,t = 0,0,0


######################################### Learning #########################################
print("********************************* LEARNING during {} secs *********************************".format(duree_app))
print()
index = 0
while (tps2-tps1) < duree_app:
    
    #Change of facial expression
    if (t%nbFramesIntegrated) == 0 :
        desiredOutput = [0,0,0,0,0]
        #index = randint(0, 4)
        index = (index+1)%5
        desiredOutput[index] = 1
        print("*********************************Supervision : " + expressions[index] + "*********************************  time elapsed : {}".format(tps2-tps1))
        print("*********************************Image_Label : " + learning_base[index][j] + "*********************************")
        frame = cv.imread(learning_base[index][j])
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                
        kps = kp_all_learn[index][j] #loading the keypoints for learning
        
        kp_l = []
        for pt_x,pt_y in kps:
            kp = cv.KeyPoint(pt_x, pt_y, 50) #creating key points
            kp_l.append(kp)
        
        sift = cv.SIFT_create(nbPoints)                     #create sift  with nbPoints keypoints
        kp, des = sift.detectAndCompute(frame, None)          #get points and descriptors
        
        #print("# kp, descriptors: {}".format(des.shape))
        
        #min and max have to be defined to normalize
        kp = kp_l
        inputImage = des                          #add descriptors matrix
    
        #Normalization of the inputs image (have data between 0 and 1)
        for j in range(nbFeatures) : 
            for i in range(len(inputImage)) :
                    inputImage[i][j] = inputImage[i][j] / 360    #normalize 
                    
        ######################################### RECOGNITION #########################################
        
    #If this not the first frame
    if t != 0 :
        
        stmImage.clearNetwork()                      #clear memory value
        
        #for each points of interest
        for point in range(nbPoints) :
            #load the data
            sawImage.load_input(inputImage[point])
    
            #Calculate saw's activity
            sawImage.calculate_nets()
            sawImage.calculate_average()
            sawImage.calculate_standard_deviation()
            sawImage.calculate_activities()#heavyside
    
            #Calculate lms's activity
            inputVector = createInputVector(sawImage.neurons, size=sawSizeImage)
            lmsImage.calculate_weightings(inputVector)
            lmsImage.calculate_activities() #sigmoide
            lmsImage.calculate_errors(desiredOutput)
            
            #Integrate activities
            inputVector = createInputVector(lmsImage.neurons, size=5)
            stmImage.integrate(inputVector)
            
            #Update weights
            
            lmsImage.update_neurons()
            sawImage.update_neuron()
       
        output = multiplyLists(inputVector, stmImage.neurons)
        print(output)
        print("Number of neuron recruits image : ",sawImage.nb_neurons)
            
            
        ######################################### DISPLAY and SAVE #########################################
        
        #cv.imshow('Frame',frame)                                                             #display the current frame
        frame=cv.drawKeypoints(frame,kp, 0,color=(0,255,255), flags=0)
        cv.imshow('Frame',frame)

    if cv.waitKey(1) & 0xFF == ord('q'):                                                 #wait to visualize
        print("\nApprentissage stopped\n")
        break
    t+=1
    index = t%5
    j = (t//5)%len(learning_base[0])
    tps2 = time.perf_counter()
    print("time : %.3f, nb_images : %d" %((tps2-tps1),(5*j+index)))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()

print(t, " iterations")

tps1 = time.perf_counter()


######################################### Test #########################################
print("********************************* Starting TEST... *********************************")
# print(3)
# time.sleep(1)
# print(2)
# time.sleep(1)
# print(1)
# time.sleep(1)
tps1 = time.perf_counter()

index,j,t = 0,0,0
len_test_base = min(len(test_base[0]),len(test_base[1]),len(test_base[2]), len(test_base[3]), len(test_base[4]))

#for performance comaprison
y_true = []        
y_pred = []

while 1 :
    
    #Get video frames
    # ret, frame = cap.read()                                 #get the image
    # current = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)         #put it in grey level
    frame = cv.imread(test_base[index][j])
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = test_base[index][j]
    img = img.split('_')[0]
    y_true.append(img) #fro the true label
    
    kps = kp_all_test[index][j] #loading the keypoints for testing
    
    kp_t = []
    for pt_x,pt_y in kps:
        kp = cv.KeyPoint(pt_x, pt_y, 10)
        kp_t.append(kp)

    
    sift = cv.SIFT_create(nbPoints)                     #create sift  with nbPoints keypoints
    kp, des = sift.detectAndCompute(frame, None)          #get points and descriptors
    
    kp = kp_t
    inputImage = des                          #add descriptors matrix
    
    #Normalization of the inputs image (have data between 0 and 1)
    for j in range(nbFeatures) : 
        for i in range(len(inputImage)) :
                inputImage[i][j] = inputImage[i][j] / 360    #normalize     

        
    stmImage.clearNetwork()                      #clear memory value
    #for each points of interest
    for point in range(nbPoints) :
        #load the data
        sawImage.load_input(inputImage[point])

        #Calculate saw's activity
        sawImage.calculate_nets()
        sawImage.calculate_average()
        sawImage.calculate_standard_deviation()
        sawImage.calculate_activities() #heavyside

        #Calculate lms's activity
        inputVector = createInputVector(sawImage.neurons, size=sawSizeImage)
        lmsImage.calculate_weightings(inputVector)
        lmsImage.calculate_activities() #sigmoide
        
        #Integrate activities
        inputVector = createInputVector(lmsImage.neurons, size=5)
        stmImage.integrate(inputVector)
        
    stmSliding.slide(stmImage.neurons)
    output = multiplyLists(inputVector, stmImage.neurons)
        
    #print("Recognition : " + expressions[np.argmax(output)])
    print("{}   {}".format(output,expressions[np.argmax(output)]) )
    output_str = expressions[np.argmax(output)]
    y_pred.append(output_str)
    ######################################### DISPLAY #########################################
    
    # output = [float('%.3f'%i) for i in output]
    # output_exp = [expressions[i] if (i%5)==np.argmax(output[5*(i//5):5*(i//5+1)]) else 0 for i in range(5)]
    # output_str = ', '.join(['%s'%i for i in output_exp if i!=0])

    # print(output)
    # print(output_exp)
    # print()
    
    #cv.imshow('Frame',frame)                                                             #display the current frame
    frame=cv.drawKeypoints(frame,kp, 0,color=(0,255,255), flags=0)
    output_str = expressions[np.argmax(output)]
    cv.putText(frame, output_str, (215,75), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.FILLED)
    cv.imshow('Frame',frame)
    
    
    time.sleep(2)
    #print("index: %d , j: %d"%(index,j))
    t+=1
    index = t%5
    j = (t//5)%len_test_base
    tps2 = time.perf_counter()
    print("time : %.3f, nb_images : %d" %((tps2-tps1),(5*j+index)))
    
    if cv.waitKey(1) & 0xFF == ord('q'):                                              #wait to visualize
        print("\nTest stopped\n")
        break

    
tps2 = time.perf_counter()
print(tps2 - tps1,"temps de test écoulé\n")


#Close video capture
# cap.release()
cv.destroyAllWindows()

abv_true = make_abv(y_true)# creating abbrevations of expression for comparison
abv_pred = make_abv(y_pred)

print(metrics.confusion_matrix(abv_true, abv_pred))
print(metrics.classification_report(abv_true, abv_pred, digits=3)) #to get the precision, recall, f1-score, accuracy 
print("Total no.of correct predictions: " + str(accuracy_score(abv_true, abv_pred, normalize=False)) + "/" + str(len(y_pred))) #for the success rate check