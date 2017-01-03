# -*- coding: utf-8 -*-

import numpy
import jieba
import jieba.posseg as pseg
import jieba.analyse
import csv
from pprint import pprint
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as ss


jieba.load_userdict('user_define.txt', )
jieba.analyse.set_stop_words('stopwords3.txt')

#File loading
stopwords = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/stopwords.txt", 'r', encoding=('utf8')).read()
data_document = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/ref_text.txt", 'r', encoding=('utf8')).read()
trainList = list(csv.reader(open('train.csv', 'r', encoding=('utf8'))))
testList = list(csv.reader(open('test.csv', 'r', encoding=('utf8'))))



#####-----Function Area-----#######
def Element2Keys(inputSentence, mode):
    output = []
    Keys = jieba.cut(inputSentence, cut_all=mode, HMM= True)
    #jieba.analyse.extract_tags(inputSentence, topK=500, withWeight=False, allowPOS=())
    for item in Keys:
        if item not in stopwords and not item.isdigit():
            output.append(item)
    return output

def most_common(lst):
    return max(set(lst), key=lst.count)
    

def expandQuery(queryKey, Dl,expandnumber):
    queryKeyEX =[]
    Dl2= ' '.join(Dl)
    Dllist = Dl2.split(' ')
    Keys = list(set(Dllist))
    #print("Dllist :", Dllist)
    #print("Keys :", Keys)
    tfarray = numpy.zeros((len(Keys),len(Dl)))
    for i in range(len(Keys)):
        if Keys[i]==queryKey:
            index = i
        for j in range(len(Dl)):
            if Keys[i] in Dl[j]:
                #print("find key", i,"at D", j)
                tfarray[i][j]+=1
            else:
                tfarray[i][j]+=0
    Cuv = tfarray.dot(tfarray.T)
    Cuvp = Cuv[index,:]
    indexlist=numpy.argsort(Cuvp,axis=0)[::-1][:expandnumber] #選top N
    for i in range(len(indexlist)):
        queryKeyEX.append(Keys[indexlist[i]])
    return queryKeyEX
    #print("queryKeyEX = \n", queryKeyEX)
def expandQuery2(queryKey, Dl,expandnumber):
    queryKeyEX =[]
    Dl2= ' '.join(Dl)
    Dllist = Dl2.split(' ')
    Keys = list(set(Dllist))
    #print("Dllist :", Dllist)
    #print("Keys :", Keys)
    tfarray = numpy.zeros((len(Keys),len(Dl)))
    #print("QueryKey in exp = ", queryKey)
    for i in range(len(Keys)):
        if Keys[i]==queryKey[0]:
            index0 = i
            #print(index0)
        elif Keys[i]==queryKey[1]:
            index1 = i
            #print(index1)
        for j in range(len(Dl)):
            if Keys[i] in Dl[j]:
                #print("find key", i,"at D", j)
                tfarray[i][j]+=1
            else:
                tfarray[i][j]+=0
    Cuv = tfarray.dot(tfarray.T)
    Cuv0 = Cuv[index0,:]
    Cuv1 = Cuv[index1,:]
    Cuvp = numpy.add(Cuv0, Cuv1)
    #print("shape",Cuv0.shape, Cuv1.shape, Cuvp.shape)
    indexlist=numpy.argsort(Cuvp,axis=0)[::-1][:expandnumber] #選top N
    for i in range(len(indexlist)):
        queryKeyEX.append(Keys[indexlist[i]])
    return queryKeyEX    
#####-----Function Area End-----#####





corpus = data_document.split('\n')

typelist = ['n','ng','nr','nrfg','nrt','ns','nt','nz','v','vd','vg','vi','vn','vq']

corpus_selected = corpus[:]
corpusList = []
for i in tqdm(range(len(corpus_selected))):
    corpusKey = Element2Keys(corpus[i], False)
    corpusList.append(corpusKey)

corpusListforTrain = corpusList
trainList.pop(0) #抽掉標題
testList.pop(0)

catlist = ['spouse','parent','child','sibling','birthPlace','deathPlace','workPlace']
categories = {}
expandnumber = 5
NotFoundPairList = []
FirstSearchDoc = []
dealDoc = 0
for item in tqdm(trainList[:]):
    keyA = item[1]
    keyB = item[2]
    categories.setdefault(item[3],[])    #print("item = ", item)
    LocalDocument = []
    findPairs = 0
    FirstSearchDoc = []
    for corpusD in corpusListforTrain:
        if (keyA in corpusD) and (keyB in corpusD):
            #print("Find A&B")
            categories[item[3]].append(corpusD) #在某一個分類中，把Doc丟進去
            FirstSearchDoc.append(' '.join(corpusD))
            corpusListforTrain.pop(corpusListforTrain.index(corpusD))
            findPairs=1
    if len(FirstSearchDoc)!=0:
        EXKey = expandQuery2(item[1:3],FirstSearchDoc, 15)
        for newkey in EXKey:
            for corpusD in corpusListforTrain:
                if newkey in corpusD:
                    categories[item[3]].append(corpusD) #在某一個分類中，把Doc丟進去
                    corpusListforTrain.pop(corpusListforTrain.index(corpusD))
                    findPairs=1
    if findPairs == 0:
        NotFoundPairList.append(item) #找不到的train成一個新的list之後expand再繼續找
    #print("EXKey = \n", EXKey)
    
print("left doc :", len(corpusListforTrain))
print("not find test pair :",len(NotFoundPairList))
#Expansion for query
indlist = []
findPairsByEXQ=0
#while len(corpusListforTrain)-len(indlist)>0:
catedDoc = 0
#print("corpuslist before expansion", len(corpusListforTrain))
while len(corpusListforTrain)>0 and expandnumber<=30:
    for items in tqdm(NotFoundPairList):
        keyAN = items[1]
        keyBN = items[2]
        relation = items[3]
        LocalDocListA = []
        LocalDocListB = []
        for Doc in corpusListforTrain:
            if keyAN in Doc:
                LocalDocListA.append(' '.join(Doc))
            elif keyBN in Doc:
                LocalDocListB.append(' '.join(Doc))
            else:
                pass
        if len(LocalDocListA)!=0 and len(LocalDocListB)!=0 and len(corpusList)!=0:
            #print("find in AEX and BEX")
            KeyAN_EX = expandQuery(keyAN,LocalDocListA,expandnumber)
            KeyBN_EX = expandQuery(keyBN,LocalDocListB,expandnumber)
            #print("KeyA EX",'\n',KeyAN_EX)
            #print("KeyB EX",'\n',KeyBN_EX)
            LocalDocFromEXA=[]
            LocalDocFromEXB=[]
            findA = 0
            findB = 0
            for Doc in corpusListforTrain:
                for i in range(min(len(KeyAN_EX),len(KeyBN_EX))):
                    if KeyAN_EX[i] in Doc:
                        findA+=1
                    if KeyBN_EX[i] in Doc:
                        findB+=1
                if findA!=0 and findB!=0:
                    categories[relation].append(Doc)
                    corpusListforTrain.pop(corpusListforTrain.index(Doc))
                    catedDoc+=1
        elif len(LocalDocListA)!=0 and len(LocalDocListB)==0 and len(corpusList)!=0:
            KeyAN_EX = expandQuery(keyAN,LocalDocListA,expandnumber)
            for Doc in corpusListforTrain:
                if keyAN in Doc:
                    categories[relation].append(Doc)
                    corpusListforTrain.pop(corpusListforTrain.index(Doc))
                    catedDoc+=1
                else:
                    pass
            #print(LocalDocFromEXA)
        elif len(LocalDocListA)==0 and len(LocalDocListB)!=0 and len(corpusList)!=0:
            KeyBN_EX = expandQuery(keyBN,LocalDocListB,expandnumber)
            for Doc in corpusListforTrain:
                if keyBN in Doc:
                    categories[relation].append(Doc)
                    corpusListforTrain.pop(corpusListforTrain.index(Doc))
                    catedDoc+=1
                else:
                    pass
            #print(LocalDocFromEXB)
        else:
            #print("Two Key not found in any docs")
            pass
    #print(len(corpusListforTrain), "Done Doc= ", catedDoc)
    if len(corpusListforTrain)!=0:
        expandnumber+=3

###-----我是分隔線，上面已經分好類了，沒分到的我只好先當沒看到...-----###
corpusListforTest = corpusList
testRelation = []
for item in testList[:1]:
    testKeyA = item[1]
    testKeyB = item[2]
    scorelist =[]
    
    #Score 1
    for i in range(len(catlist)):
        score = 0
        for doc in categories[catlist[i]]:
            #print(len(categories[catlist[i]]))
            if (testKeyA in doc) and (testKeyB in doc):
                score+=2
            elif (testKeyA in doc) and (testKeyB not in doc):
                score+=1
            elif (testKeyA not in doc) and (testKeyB in doc):
                score+=1
            else:
                score+=0
        scorelist.append(score)
    '''#Score 2
    LocalDocListTestA = []
    LocalDocListTestB = []
    LocalDocListTestAB = []
    for Doc in corpusListforTest:
        if (testKeyA in Doc) and (testKeyB in Doc):
            LocalDocListTestAB.append(' '.join(Doc))
        elif testKeyA in Doc:
            LocalDocListTestA.append(' '.join(Doc))
        elif testKeyB in Doc:
            LocalDocListTestB.append(' '.join(Doc))
        else:
            pass    
    if len(LocalDocListTestAB)!=0:
        TestKeyAB_EX = expandQuery2(item[1:3],LocalDocListTestAB,10)
    if len(LocalDocListTestA)!=0:
        TestKeyA_EX = expandQuery2(item[1],LocalDocListTestA,10)
    if len(LocalDocListTestB)!=0:
        TestKeyB_EX = expandQuery2(item[3],LocalDocListTestB,10)     
    '''
    max_value = max(scorelist)
    max_index = scorelist.index(max_value)
    testRelation.append(catlist[max_index])
    
    