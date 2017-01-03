# -*- coding: utf-8 -*-

import csv
import pickle
import numpy
from tqdm import *
import jieba.posseg as pseg
#----------------Fuctions----------------#
def expandQuery(queryKey, Dl):
    queryKeyEX =[]
    Dl2=[]
    qKey2 = []
    Keys=[]
    for docs in Dl:
        for doc in docs:
            Dl2.append(doc)
    for ks in queryKey:
        qKey2.append(ks)
    #print("query",qKey2)
    #print("Dl2",Dl2)
    #Dllist = Dl2.split(' ')
    #Dl3 = [item for sublist in Dl2 for item in sublist]
    Keys0 = list(set(Dl2))
    #print("Dl3: ",Dl3)
    #queryKey = [item for sublist in qKey2 for item in sublist]
    queryKey0 = list(set(qKey2))
    #print("Keys0 ", Keys0)
    for key in Keys0:
        if key in queryKey0 or len(key)>1:
            Keys.append(key)
    #print("queryKey0 ", queryKey0)
    #print("Dj ", len(Dl))
    tfarray = numpy.zeros((len(Keys),len(Dl)))
    Cuvparray = numpy.zeros((len(Keys),len(Keys)))
    indexOfQuery = []
    for i in range(len(Keys)):
        if Keys[i] in queryKey0:
            #print("find Key in Query")
            indexOfQuery.append(i)
        for j in range(len(Dl)):
            if Keys[i] in Dl[j]:
                tfarray[i][j]+=1
            else:
                tfarray[i][j]+=0
    #print("len(Keys)", len(Keys0))
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

#-----------------------------------------------------------------------------#
Dictionary = pickle.load(open("Dictionary1226.txt",'rb'))
corpus = pickle.load(open("Corpus1225.txt",'rb'))
testList = list(csv.reader(open("test_fix.csv", 'r', encoding='utf8')))
trainList = list(csv.reader(open("train_fix.csv", 'r', encoding='utf8')))
scorelist = {}
typebaseline={}
sumthenumber=0
for k, v in Dictionary.items():
    sumthenumber=sum(len(v) for v in Dictionary.values())
    typebaseline.setdefault(k,[])
    typebaseline[k]=round(len(Dictionary[k])/sumthenumber,4)
for k, v in typebaseline.items():
    typebaseline[k]=round(v*100/sumthenumber,4)
count = 0
score = 0
relation = []
idList=[]
for p in tqdm(trainList[200:210], desc="Model 執行..."):
    #print(p)
    ID = p[0]
    idList.append(ID)
    pairs = p[1:3] 
    LocalCorpus = []
    for doc in corpus:
        if any(x in doc for x in pairs):
            LocalCorpus.append(doc)
    if LocalCorpus:
        count+=1
        for k in Dictionary.keys():
            scorelist.setdefault(k, [])
            scorelist[k]=0
        extractKeys = expandQuery(pairs, LocalCorpus)
        for key in extractKeys:
            #print(key[0])
            for R in Dictionary.keys():#按順序一個一個字典查（之後可以考慮先按詞性分再查）
                for W in Dictionary[R]:
                    for w in W:
                        #print(w, w[0])
                        if key[0]==w[0]:
                            #print("Match")
                            scorelist[R] += w[1]#*w[1]
        tempKey = pseg.lcut(p[2])
        if list(tempKey[0])[1] in ['nr']:
            for r in ['parent', 'child', 'spouse', 'sibling']:
                scorelist[r]=scorelist[r]*2
        elif list(tempKey[0])[1] in ['ns']:
            for r in ['birthPlace', 'deathPlace', 'workPlace']:
                scorelist[r]=scorelist[r]*2 
        EstimateResult =max(scorelist, key=(lambda key:scorelist[key]))

    while EstimateResult!=p[3]:
        print("refine-ing")
        for key in extractKeys:
            #print(key[0])
            for W in Dictionary[p[3]]:
                for w in W:
                    if key[0]==w[0]:
                        w[1]=w[1]+key[1]
        for key in extractKeys:
            for k in Dictionary.keys():
                scorelist[k]=0
            #print(key[0])
            for R in Dictionary.keys():#按順序一個一個字典查（之後可以考慮先按詞性分再查）
                for W in Dictionary[R]:
                    for w in W:
                        #print(w, w[0])
                        if key[0]==w[0]:
                            #print("Match")
                            scorelist[R] += w[1]
        EstimateResult =max(scorelist, key=(lambda key:scorelist[key]))

with open("Dictionary1226Refine.txt", "wb") as Dp:
    pickle.dump(Dictionary, Dp)



'''if EstimateResult==p[3]:
    score+=1
else:
    print("預測值:", EstimateResult, '\t', "正確答案：", p[3])
    for k, v in scorelist.items():
        print(k, v)
print("準確率=", round(score*100/count, 2),"%")'''


