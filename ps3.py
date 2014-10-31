# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 17:29:53 2014
@author: jascha, tasso
"""

import os
import csv

path = './PS3_dev/'
ratio = 0.80

files = os.listdir(path)

# List of tokens that can be removed for first task
markupTokens = {'{sl}', 'sp', '{ls}', '{lg}', '{cg}', '{ns}',
                '{br}', '*', '[', ']'}

# List of question words used to detect a question phrase
questionWords = {'why', 'what', 'how', 'when' ,'where', 'is'}

full_set = []
for file in files:
    f = open(path + file,'rb')
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        full_set.append(row)

train_set = full_set[:int(len(full_set)*ratio)]
test_set = full_set[int(len(full_set)*ratio):]

# List to hold train_set list with tokens removed
cleanedTrain = []

# List to hold the Q/A for each line
trainResults = []

i = 0

# First task: decide on question/answer
for line in train_set:
    
    currentPhrase = line[5]
    
    # Clean up each line by removing tokens
    for token in markupTokens:
        if token in currentPhrase:
            currentPhrase = currentPhrase.replace(token, '')
            
    cleanedTrain.append(currentPhrase)
            
    firstThreeWords = currentPhrase.split()[:3]
    print firstThreeWords
    
    # Assume the current line is an answer
    trainResults.append('A')
    
    # If a question word is found in the first two words of
    # the line, change the line to a question
    for q in questionWords:
        if q in firstThreeWords:
            trainResults[i] = 'Q'
            break
        
    i += 1

numCorrect = 0

for i in range(len(trainResults)):
    if trainResults[i] == train_set[i][3]:
        #print train_set[i][5] + trainResults[i]
        numCorrect += 1

print str(numCorrect) + " correct assignments"
print str((float(numCorrect)/len(train_set)) * 100) + "% accuracy"