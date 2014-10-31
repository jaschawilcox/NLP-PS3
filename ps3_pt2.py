# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 17:29:53 2014
@author: jascha, tasso
"""

import os
import csv

path = './data/' # Path to csv files
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

