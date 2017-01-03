# -*- coding: utf-8 -*-

import csv
import pickle
import numpy

import jieba.posseg as pseg
from tqdm import *

#----------------Fuctions----------------#
def expandQuery(queryKey, Dl):
    #print(len(Dl))
    queryKeyEX =[]
    Dl2=[]
    qKey2 = []
    Keys=[]
    for docs in Dl:
        for doc in docs:
            #print("doc:", doc)
            Dl2.append(doc)
    #print("query",qKey2)
    #print("Dl2",Dl2[0])
    #Dllist = Dl2.split(' ')
    Dl3 = [item for sublist in Dl2 for item in sublist]
    Keys0 = list(set(Dl3))
    #print("Dl3: ",Dl3)
    queryKey = [item for sublist in queryKey for item in sublist]
    queryKey0 = list(set(queryKey))
    #print("queryKey0 ", queryKey0)
    for key in Keys0:
        if len(key)>1:
            Keys.append(key)
    #print("Keys ", len(Keys))
    #print("Dj ", len(Dl))
    tfarray = numpy.zeros((len(Keys),len(Dl2)))
    Cuvparray = numpy.zeros((len(Keys),len(Keys)))
    indexOfQuery = []
    for i in range(len(Keys)):
        if Keys[i] in queryKey0:
            #print("find Key in Query")
            indexOfQuery.append(i)
        for j in range(len(Dl2)):
            if Keys[i] in Dl2[j]:
                #print("find in ", Dl2[j])
                tfarray[i][j]+=1
            else:
                tfarray[i][j]+=0
    #print("indexOfQuery",len(indexOfQuery))
    Cuv = tfarray.dot(tfarray.T)
    #print("Cuv=\n", Cuv)
    for i in range(len(Keys)):
        for j in range(len(Keys)):
            if Cuv[i][j]!=0:
                Cuvparray[i][j]=Cuv[i][j]/(Cuv[i][i]+Cuv[j][j]-Cuv[i][j])
    #print("CuvP=\n", Cuvparray)
    Cuv0 = numpy.zeros((len(Keys),))
    for x in indexOfQuery:
        if indexOfQuery.index(x)==0:
            Cuvp = numpy.add(Cuv0,Cuvparray[x,:])
        else:
            Cuvp = numpy.add(Cuvp,Cuvparray[x,:])
    #print("Cuvp", Cuvp)
    #planB
    #Cuvp=numpy.zeros((len(Keys),))
    #print(Cuvp)
    indexlist2=numpy.argsort(Cuvp,axis=0)[::-1] #選top N
    for i in range(len(indexlist2)):
        resultlist = []
        if Cuvp[indexlist2[i]] and Keys[indexlist2[i]] not in queryKey:
            resultlist.append(Keys[indexlist2[i]])
            resultlist.append(Cuvp[indexlist2[i]])#/Cuvp[indexlist2[0]])
            queryKeyEX.append(resultlist)
    #print("queryKeyEX", queryKeyEX[:20])
    return queryKeyEX

def expandQuery2(queryKey, Dl): #移除人名
    #print(len(Dl))
    queryKeyEX =[]
    Dl2=[]
    qKey2 = []
    Keys=[]
    for docs in Dl:
        for doc in docs:
            #print("doc:", doc)
            Dl2.append(doc)
    #print("query",qKey2)
    #print("Dl2",Dl2[0])
    #Dllist = Dl2.split(' ')
    Dl3 = [item for sublist in Dl2 for item in sublist]
    Keys0 = list(set(Dl3))
    #print("Dl3: ",Dl3)
    queryKey = [item for sublist in queryKey for item in sublist]
    queryKey0 = list(set(queryKey))
    #print("queryKey0 ", queryKey0)
    for key in Keys0:
        tempKey = pseg.lcut(key)
        #print(list(tempKey[0]))
        if key not in queryKey0 and len(key)>1 and list(tempKey[0])[1] not in ['nr']:
            Keys.append(key)
    #print("Keys ", len(Keys))
    #print("Dj ", len(Dl))
    tfarray = numpy.zeros((len(Keys),len(Dl2)))
    Cuvparray = numpy.zeros((len(Keys),len(Keys)))
    indexOfQuery = []
    for i in range(len(Keys)):
        if Keys[i] in queryKey0:
            #print("find Key in Query")
            indexOfQuery.append(i)
        for j in range(len(Dl2)):
            if Keys[i] in Dl2[j]:
                #print("find in ", Dl2[j])
                tfarray[i][j]+=1
            else:
                tfarray[i][j]+=0
    #print("indexOfQuery",len(indexOfQuery))
    Cuv = tfarray.dot(tfarray.T)
    #print("Cuv=\n", Cuv)
    for i in range(len(Keys)):
        for j in range(len(Keys)):
            if Cuv[i][j]!=0:
                Cuvparray[i][j]=Cuv[i][j]/(Cuv[i][i]+Cuv[j][j]-Cuv[i][j])
    #print("CuvP=\n", Cuvparray)
    Cuv0 = numpy.zeros((len(Keys),))
    for x in indexOfQuery:
        if indexOfQuery.index(x)==0:
            Cuvp = numpy.add(Cuv0,Cuvparray[x,:])
        else:
            Cuvp = numpy.add(Cuvp,Cuvparray[x,:])
    #print("Cuvp", Cuvp)
    #planB
    #Cuvp=numpy.zeros((len(Keys),))
    #print(Cuvp)
    indexlist2=numpy.argsort(Cuvp,axis=0)[::-1] #選top N
    for i in range(len(indexlist2[:(len(Dl2)*10)])):
        resultlist = []
        if Cuvp[indexlist2[i]] and Keys[indexlist2[i]] not in queryKey:
            resultlist.append(Keys[indexlist2[i]])
            resultlist.append(Cuvp[indexlist2[i]])#/Cuvp[indexlist2[0]])
            queryKeyEX.append(resultlist)
    #print("queryKeyEX", queryKeyEX[:20])
    return queryKeyEX
#----------------Read File----------------#
trainList = list(csv.reader(open("train_fix.csv", 'r', encoding='utf8')))
corpus = pickle.load(open("Corpus1225.txt",'rb'))
Vocabulary = {}
#print(trainList[1])
'''
for p in tqdm(trainList[1:]):
	ID = p[0]
	pairs = p[1:3]
	R = p[3]
	Vocabulary.setdefault(R, [])
	Vocabulary[R].append(pairs)

with open("Vocabulary1225.txt", "wb") as Vp:
    pickle.dump(Vocabulary, Vp)'''
#print(Vocabulary['child'])

#print((Vocabulary['child']))
'''
CORPUS = {}
for R in tqdm(Vocabulary.keys()):
	LocalCorpus=[]
	for pairs in Vocabulary[R]:
		for Doc in corpus:
			if all(x in Doc for x in pairs):
				LocalCorpus.append(Doc)
	CORPUS.setdefault(R, [])
	CORPUS[R].append(LocalCorpus)
#print(CORPUS.keys())
with open("CORPUSbyRelation1225.txt", "wb") as Rp:
    pickle.dump(CORPUS, Rp)'''
#Vocabulary = pickle.load(open("Vocabulary1225.txt",'rb'))
CORPUS = pickle.load(open("CORPUSbyRelation1225.txt",'rb'))
#Dictionary = pickle.load(open("Dictionary1225.txt",'rb'))
'''
Dictionary = {}
for R in tqdm(CORPUS.keys()):
    #print(Vocabulary[R])
	Dictionary.setdefault(R, [])
	words = expandQuery2(Vocabulary[R], CORPUS[R])
	Dictionary[R].append(words)
#for k, v in Dictionary.items():
#    print( k, v)



with open("Dictionary1226Compress.txt", "wb") as Dp:
    pickle.dump(Dictionary, Dp)'''

