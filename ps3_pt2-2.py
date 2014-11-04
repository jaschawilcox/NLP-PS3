# -*- coding: utf-8 -*-
"""
Created on Mon Nov 03 09:22:03 2014

@author: aditya, jacha, tasso
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import metrics
#from operator import itemgetter
#from sklearn.metrics import classification_report
import os
import csv
#import pandas
#from gensim import corpora, models, similarities
#from itertools import chain
#import nltk
#from nltk.corpus import stopwords
#from operator import itemgetter
#import re

path = './data/' # Path to csv files
testPath = './testdata/' # Path to final test csv files
ratio = 0.80

files = os.listdir(path)
files = [f for f in files if f.find('.csv') > 0]

testFiles = os.listdir(testPath)
testFiles = [f for f in testFiles if f.find('.csv') > 0]

# List of tokens that can be removed for first task
markupTokens = {'{sl}', 'sp', '{ls}', '{lg}', '{cg}', '{ns}',
                '{br}', '*', '[', ']'}

# List of question words used to detect a question phrase
questionWords = {'why', 'what', 'how', 'when' ,'where', 'is', 'do',
                 'what\'s'}

full_set = []
for file in files:
    f = open(path + file,'rb')
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        full_set.append(row)
        
# Held-out data set
finalTestSet = []
for file in testFiles:
    f = open(path + file,'rb')
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        finalTestSet.append(row)
        
# Lists used for classification in task 2
X_train = []
y_train = []
X_test= []
y_test = []
finalTest = []
finalTestTruth = []
train_set = full_set[:int(len(full_set)*ratio)]
test_set = full_set[int(len(full_set)*ratio):]

# List to hold train_set list with tokens removed
cleanedTrain = []

# List to hold the Q/A for each line
trainResults = []
finalQAResults = []

# List to hold the E/M for each line
finalEMResults = []

# Used for next for loop
i = 0

# First task: decide on question/answer
for line in full_set:
    
    currentPhrase = line[5]
    
    # Clean up each line by removing tokens
    for token in markupTokens:
        if token in currentPhrase:
            currentPhrase = currentPhrase.replace(token, '')
            
    cleanedTrain.append(currentPhrase)
            
    firstThreeWords = currentPhrase.split()[:3]
#    print firstThreeWords
    
    # Assume the current line is an answer
    trainResults.append('A')
    
    # If a question word is found in the first three words of
    # the line, change the line to a question
    for q in questionWords:
        if q in firstThreeWords:
            trainResults[i] = 'Q'
            break
        
    if (len(currentPhrase.split()) > 20):
        trainResults[i] = 'A'
        
    i += 1

numCorrect = 0

# Determine correctness in training set by comparing answer array
# with actual values from file
for i in range(len(trainResults)):
    if trainResults[i] == full_set[i][3]:
        #print train_set[i][5] + trainResults[i]
        numCorrect += 1
    else:
        print full_set[i]

print str(numCorrect) + " correct assignments"
print str((float(numCorrect)/len(full_set)) * 100) + "% accuracy"

i = 0

for line in finalTestSet:
    
    currentPhrase = line[5]
    
    # Clean up each line by removing tokens
    for token in markupTokens:
        if token in currentPhrase:
            currentPhrase = currentPhrase.replace(token, '')
            
    firstThreeWords = currentPhrase.split()[:3]
#    print firstThreeWords
    
    # Assume the current line is an answer
    finalQAResults.append('A')
    
    # If a question word is found in the first three words of
    # the line, change the line to a question
    for q in questionWords:
        if q in firstThreeWords:
            finalQAResults[i] = 'Q'
            break
        
    if (len(currentPhrase.split()) > 20):
        finalQAResults[i] = 'A'
        
    i += 1

############# Beginning of E/M classification #################
for line in train_set:
    
    #currentTrainFeature = line[5]
    currentTrainFeature = ' '.join((line[0], line[1], line[2], line[5]))
    currentTrainGT = line[4]
    X_train.append(currentTrainFeature)
    y_train.append(currentTrainGT)
    
for line in test_set:
    
    #currentTestFeature = line[5]
    currentTestFeature = ' '.join((line[0], line[1], line[2], line[5]))
    currentTestGT = line[4]
    X_test.append(currentTestFeature)
    y_test.append(currentTestGT)
    
for line in finalTestSet:
    
    currentFTestFeature = ' '.join((line[0], line[1], line[2], line[5]))
    currentFTestGT = line[4]
    finalTest.append(currentFTestFeature)
    finalTestTruth.append(currentFTestGT)

#Convert Ground Truth To Array for use with the SKLEARN Metrics MODULE

import numpy as np
testGTarray = np.asarray(y_test)

# Multinomial Naive Bayes
vectorizer = TfidfVectorizer(min_df=2, 
 ngram_range=(1, 2), 
 stop_words='english', 
 strip_accents='unicode', 
 norm='l2')
 
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
finalTest = vectorizer.transform(finalTest)
nb_classifier = MultinomialNB().fit(X_train, y_train)
y_nb_predicted = nb_classifier.predict(X_test)
finalNBPredicted = nb_classifier.predict(finalTest)

#print "MODEL: Multinomial Naive Bayes\n"
#
#print 'The precision for this classifier is ' + str(metrics.precision_score(testGTarray, y_nb_predicted))
#print 'The recall for this classifier is ' + str(metrics.recall_score(testGTarray, y_nb_predicted))
#print 'The f1 for this classifier is ' + str(metrics.f1_score(testGTarray, y_nb_predicted))
#print 'The accuracy for this classifier is ' + str(metrics.accuracy_score(testGTarray, y_nb_predicted))
#
#print '\nHere is the classification report:'
#print classification_report(testGTarray, y_nb_predicted)
#
##simple thing to do would be to up the n-grams to bigrams; try varying ngram_range from (1, 1) to (1, 2)
##we could also modify the vectorizer to stem or lemmatize
#print '\nHere is the confusion matrix:'
#print metrics.confusion_matrix(testGTarray, y_nb_predicted)    
    
# SVM
from sklearn.svm import LinearSVC
svm_classifier = LinearSVC().fit(X_train, y_train)
y_svm_predicted = svm_classifier.predict(X_test)   
finalSVMPredicted = svm_classifier.predict(finalTest) 

# Logistic Regression

from sklearn.linear_model import LogisticRegression
maxent_classifier = LogisticRegression().fit(X_train, y_train)
y_maxent_predicted = maxent_classifier.predict(X_test)
finalMaxEntPredicted = maxent_classifier.predict(finalTest)

## Unsupervised Topic Model Based Ckustering
#
#documents = X_train
#stoplist = stopwords.words('english')
#texts = [[word for word in document.lower().split() if word not in stoplist]
# for document in documents]
#
#dictionary = corpora.Dictionary(texts)
#corpus = [dictionary.doc2bow(text) for text in texts]
#
#tfidf = models.TfidfModel(corpus) 
#corpus_tfidf = tfidf[corpus]
#
##lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
##lsi.print_topics(20)
#
#n_topics = 2
#lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
#
#for i in range(0, n_topics):
# temp = lda.show_topic(i, 10)
# terms = []
# for term in temp:
#     terms.append(term[1])
# print "Top 10 terms for topic #" + str(i) + ": "+ ", ".join(terms)
# 
#print 
#print 'Which LDA topic maximally describes a document?\n'
#print 'Original document: ' + documents[1]
#print 'Preprocessed document: ' + str(texts[1])
#print 'Matrix Market format: ' + str(corpus[1])
#print 'Topic probability mixture: ' + str(lda[corpus[1]])
#print 'Maximally probable topic: topic #' + str(max(lda[corpus[1]],key=itemgetter(1))[0])
    
#y_train = np.array([el for el in nyt_labels[0:trainset_size]])
#
#X_test = np.array([''.join(el) for el in nyt_data[trainset_size+1:len(nyt_data)]]) 
#y_test = np.array([el for el in nyt_labels[trainset_size+1:len(nyt_labels)]]) 

# Get the majority prediction from the 3 classifiers above for each line
# and use it as the classification
j = 0

for truth in finalTestTruth:
    numM = 0
    numE = 0
    currentPredictions = [finalNBPredicted[j], finalSVMPredicted[j],
                   finalMaxEntPredicted[j]]
    for prediction in currentPredictions:
        if prediction == 'M':
            numM += 1
        else: numE += 1
    if numM > 1:
        bestPrediction = 'M'
    else: bestPrediction = 'E'
    finalEMResults.append(bestPrediction)
    
    #print currentPredictions, bestPrediction
    j += 1

# Write results out to a .csv file in the same format as input
if not os.path.exists('./result/'):
    os.makedirs('./result/')
resultFile = open('./result/results.csv', 'w+')

j = 0

for line in finalTestSet:
    resultFile.write(line[0] + ',' + line[1] + ',' + line[2] + ',' +
            finalQAResults[j] + ',' + finalEMResults[j] +
            ',' + line[5] + '\n')
    j += 1

resultFile.close()
#print finalEMResults
#print finalQAResults