#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:30:51 2016

@author: chenyu
"""

import numpy as np
import csv
import pickle
from tqdm import *

#File loading
stopwords = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/stopwords.txt", 'r', encoding=('utf8')).read()
#data_document = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/tokenizedCorpus03.txt", 'r', encoding=('utf8')).read()
trainList = list(csv.reader(open('train.csv', 'r', encoding=('utf8'))))
testList = list(csv.reader(open('test.csv', 'r', encoding=('utf8'))))

#with open ("tokenizedCorpus_jieba.txt", "rb") as fp:
corpuslist = pickle.load(open("tokenizedCorpus_jieba.txt", "rb"))
#corpuslist00 = pickle.load(open("Corpus_wordtype_bosontrain.txt","rb"))
#-----------------------前處理-----------------------#
#先抽出word 建立一個要用的corpus

trainList.pop(0)
corpuslistForTrain = list(corpuslist)
PairListAandB=[]
PairListAorB=[]

NewCorpusAandB =[]
NewCorpusAorB =[]
for c in tqdm(range(len(corpuslistForTrain[:])),desc='Start to find'):
    found=0
    for pairs in trainList:
        if any(x in corpuslistForTrain[c] for x in pairs[1:3]):
            if all(x in corpuslistForTrain[c] for x in pairs[1:3]):
                found=2
            elif found!=2:
                found=1
    if found==2:
        NewCorpusAandB.append(corpuslistForTrain[c])
    elif found ==1:
        NewCorpusAorB.append(corpuslistForTrain[c])

for p in tqdm(trainList[:]):
    found= 0
    for doc in corpuslistForTrain:
        if any(x in doc for x in p[1:3]):
            if all(x in doc for x in p[1:3]):
                #print(p[1:3])
                found=2
            elif found!=2:
                found=1
    if found==2:
        PairListAandB.append(p)
    elif found==1:
        PairListAorB.append(p)
            
        
with open("CorpusAandBFound_J.txt", "wb") as fpAB:
    pickle.dump(NewCorpusAandB, fpAB)
with open("CorpusAorBFound_J.txt", "wb") as fpAorB:
    pickle.dump(NewCorpusAorB, fpAorB)
with open("TrainABFound_J.txt", "wb") as TAandB:
    pickle.dump(PairListAandB, TAandB)
with open("TrainAorBFound_J.txt", "wb") as TAorB:
    pickle.dump(PairListAorB, TAorB)
            