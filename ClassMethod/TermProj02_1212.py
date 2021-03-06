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
    #Keys = jieba.analyse.extract_tags(inputSentence, topK=500, withWeight=False, allowPOS=())
    for item in Keys:
        if item not in stopwords and not item.isdigit():
            output.append(item)
    return output

def most_common(lst):
    return max(set(lst), key=lst.count)
    
def expandQuery2(queryKey, Dl,expandnumber):
    queryKeyEX =[]
    Dl2= ' '.join(Dl)
    Dllist = Dl2.split(' ')
    checkA=0
    checkB=0
    Keys = list(set(Dllist))
    tfarray = numpy.zeros((len(Keys),len(Dl)))
    for i in range(len(Keys)):
        if Keys[i]==queryKey[0]:
            index0 = i
            checkA = 1
        elif Keys[i]==queryKey[1]:
            index1 = i
            checkB = 1
        for j in range(len(Dl)):
            if Keys[i] in Dl[j]:
                tfarray[i][j]+=1
            else:
                tfarray[i][j]+=0
    Cuv = tfarray.dot(tfarray.T)
    if checkA!=0:
        Cuv0 = Cuv[index0,:]
    else:
        Cuv0 = numpy.zeros((len(Keys),))
    if checkB!=0:
        Cuv1 = Cuv[index1,:]
    else:
        Cuv1 = numpy.zeros((len(Keys),))
    Cuvp = numpy.add(Cuv0, Cuv1)
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

corpusListforTrain = list(corpusList)
trainList.pop(0) #抽掉標題
testList.pop(0)

catlist = ['spouse','parent','child','sibling','birthPlace','deathPlace','workPlace']
categories = {}
expandnumber = 5
FirstSearchDoc = []
ExpSearchDoc = []
dealDoc = 0
for item in tqdm(trainList[:], desc="First Search"):
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
            FirstSearchDoc.append(corpusD)
            corpusListforTrain.pop(corpusListforTrain.index(corpusD))            
print("left doc :", len(corpusListforTrain))
#Expansion for query
indlist = []
findPairsByEXQ=0
#while len(corpusListforTrain)-len(indlist)>0:
catedDoc = 0

#選擇想要的分類方式，選1的話就是(A+B)+(A+B)'合併篩選，2的話還沒決定哈哈
Select = 2

#print("corpuslist before expansion", len(corpusListforTrain))
while expandnumber<=10:
    for items in tqdm(trainList[:], desc="Expand Search"):
        #print(trainList.index(items))
        keyA = items[1]
        keyB = items[2]
        relation = items[3]
        LocalDocListTrainA = []
        LocalDocListTrainB = []
        LocalDocListTrainAB = []
        othersDocList = []
        LocalDocListTrainABr =[]
        KeyAB_EX = []
        KeyABr_EX = []
        for Doc in corpusList:
            if (keyA in Doc) and (keyB in Doc):
                #print("Find A and B")
                LocalDocListTrainAB.append(' '.join(Doc))
            elif (keyA in Doc) and (keyB not in Doc):
                LocalDocListTrainA.append(' '.join(Doc))
            elif (keyA not in Doc) and (keyB in Doc):
                LocalDocListTrainB.append(' '.join(Doc))
            else:
                othersDocList.append(Doc)
        LocalDocListTrainABr= LocalDocListTrainA+LocalDocListTrainB
        if len(LocalDocListTrainAB)!=0:
            KeyAB_EX = expandQuery2(items[1:3], LocalDocListTrainAB, expandnumber)
        elif len(LocalDocListTrainA)!=0 or len(LocalDocListTrainB)!=0:
            KeyABr_EX = expandQuery2(items[1:3],LocalDocListTrainABr,expandnumber)
        if Select ==1:#方案1
            ExpandKey= list(set(KeyAB_EX+KeyABr_EX))
            for Docs in corpusListforTrain:
                findkey=0
                for k in ExpandKey:
                    if k in Docs:
                        findkey+=1
                if findkey>1:#找到2個以上才算該類
                    categories[relation].append(Docs)
                    ExpSearchDoc.append(Docs)
                    corpusListforTrain.pop(corpusListforTrain.index(Docs))
        elif Select ==2:
            for Docs in corpusListforTrain:
                findkey=0
                for k in KeyAB_EX:
                    if k in Docs:
                        findkey+=1
                if findkey>0:#找到2個以上才算該類
                    categories[relation].append(Docs)
                    ExpSearchDoc.append(Docs)
                    corpusListforTrain.pop(corpusListforTrain.index(Docs))
                findkey=0
                for k in KeyABr_EX:
                    if k in Docs:
                        findkey+=1
                if findkey>1:#找到2個以上才算該類
                    categories[relation].append(Docs)
                    ExpSearchDoc.append(Docs)
                    corpusListforTrain.pop(corpusListforTrain.index(Docs))
    print("已分類", len(corpusList)-len(corpusListforTrain),"篇文章", "剩下",len(corpusListforTrain),"篇尚未分類")
    if len(corpusListforTrain)!=0:
        expandnumber+=2

###-----我是分隔線，上面已經分好類了，沒分到的我只好先當沒看到...-----###
'''
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
    #Score 2
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
    max_value = max(scorelist)
    max_index = scorelist.index(max_value)
    testRelation.append(catlist[max_index])
    
    '''