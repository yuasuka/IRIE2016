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
from collections import Counter
import pickle
from queue import Queue
from threading import Thread

jieba.load_userdict('user_define.txt', )
jieba.analyse.set_stop_words('stopwords3.txt')

#File loading
stopwords = open("stopwords.txt", 'r', encoding=('utf8')).read()
data_document = open("ref_text.txt", 'r', encoding=('utf8')).read()
trainList = list(csv.reader(open('train.csv', 'r', encoding=('utf8'))))
testList = list(csv.reader(open('test.csv', 'r', encoding=('utf8'))))

#global
global TrainListQueue
TrainListQueue = Queue()


#####-----Function Area-----#######
def Element2Keys(inputSentence, mode):
    output = []
    Keys = jieba.cut(inputSentence, cut_all=mode)
    #Keys = jieba.analyse.extract_tags(inputSentence, topK=500, withWeight=False, allowPOS=())
    #Keys = jieba.analyse.textrank(inputSentence, topK=500, withWeight=False, allowPOS=())
    for item in Keys:
        if item not in stopwords and not item.isdigit():
            output.append(item)
    return output

def most_common(lst):
    return max(set(lst), key=lst.count)
    
def expandQuery2(queryKey, Dl):
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
    indexlist=numpy.argsort(Cuvp,axis=0)[::-1][:52] #選top N
    for i in range(len(indexlist)):
        queryKeyEX.append(Keys[indexlist[i]])
    if queryKey[0] in queryKeyEX:
        queryKeyEX.remove(queryKey[0])
    if queryKey[1] in queryKeyEX:
        queryKeyEX.remove(queryKey[1])
    return queryKeyEX    

#####-----Function Area End-----#####




corpus = data_document.split('\n')
typelist = ['n','ng','nr','nrfg','nrt','ns','nt','nz','v','vd','vg','vi','vn','vq']
corpus_selected = corpus[:]
corpusList = []
for i in tqdm(range(len(corpus_selected))):
    corpusKey = Element2Keys(corpus_selected[i], False)
    corpusList.append(corpusKey)

corpusListforTrain = list(corpusList)
trainList.pop(0) #抽掉標題
testList.pop(0)

catlist = ['spouse','parent','child','sibling','birthPlace','deathPlace','workPlace']
categories = {}
vocabularies = {}
FirstSearchDoc = []
ExpSearchDoc = []
dealDoc = 0
#while expandnumber<=10:
EXPANDKEYSlist = []
Relationlist = []
for x in trainList:
    TrainListQueue.put(x)
#--------------------workers--------------------#
def workers():
    items = TrainListQueue.get()
    IDnumber = items[0]
    keyA = items[1]
    keyB = items[2]
    relation = items[3]
    categories.setdefault(relation,[])    
    LocalDocListTrainA = []
    LocalDocListTrainB = []
    LocalDocListTrainAB = []
    othersDocList = []
    LocalDocListTrainABr =[]
    KeyAB_EX = []
    KeyABr_EX = []
    for Doc in corpusListforTrain:
        if (keyA in Doc) and (keyB in Doc):
            #print("Find A and B")
            LocalDocListTrainAB.append(' '.join(Doc))
            vocabularies.setdefault(relation,[])
            categories[relation].append(Doc)
            corpusListforTrain.pop(corpusListforTrain.index(Doc))
        elif (keyA in Doc) and (keyB not in Doc):
            LocalDocListTrainA.append(' '.join(Doc))
        elif (keyA not in Doc) and (keyB in Doc):
            LocalDocListTrainB.append(' '.join(Doc))
        else:
            othersDocList.append(Doc)
    #print("First Search分類移除",len(corpusList)-len(corpusListforTrain),"篇，剩下",len(corpusListforTrain),"篇")
    LocalDocListTrainABr= LocalDocListTrainA+LocalDocListTrainB
    #print("與",keyA, "、",keyB,"個別相關的文章共計",len(LocalDocListTrainABr),"篇")
    if len(LocalDocListTrainAB)!=0:
        KeyAB_EX = expandQuery2(items[1:3], LocalDocListTrainAB)
        EXPANDKEYSlist.append(KeyAB_EX)
        Relationlist.append(relation)
    elif len(LocalDocListTrainABr)!=0:
        KeyABr_EX = expandQuery2(items[1:3],LocalDocListTrainABr)
        EXPANDKEYSlist.append(KeyABr_EX)
        Relationlist.append(relation)
    else:
        EXPANDKEYSlist.append(items[1:3])
        Relationlist.append(relation)
        #print(items[1:3],relation)
    return
#--------------------workers--------------------#


t={}
for x in range(5):
    t['treads{}'.format(x)] = Thread(target=workers)
for p in t.keys():
    t[p].start()
for p in t.keys():
    t[p].join()


for items in tqdm(trainList[:], desc='Training Time'):
    #print("Train #",trainList.index(items))
    keyA = items[1]
    keyB = items[2]
    relation = items[3]
    categories.setdefault(relation,[])    
    LocalDocListTrainA = []
    LocalDocListTrainB = []
    LocalDocListTrainAB = []
    othersDocList = []
    LocalDocListTrainABr =[]
    KeyAB_EX = []
    KeyABr_EX = []
    for Doc in corpusListforTrain:
        if (keyA in Doc) and (keyB in Doc):
            #print("Find A and B")
            LocalDocListTrainAB.append(' '.join(Doc))
            vocabularies.setdefault(relation,[])
            categories[relation].append(Doc)
            corpusListforTrain.pop(corpusListforTrain.index(Doc))
        elif (keyA in Doc) and (keyB not in Doc):
            LocalDocListTrainA.append(' '.join(Doc))
        elif (keyA not in Doc) and (keyB in Doc):
            LocalDocListTrainB.append(' '.join(Doc))
        else:
            othersDocList.append(Doc)
    #print("First Search分類移除",len(corpusList)-len(corpusListforTrain),"篇，剩下",len(corpusListforTrain),"篇")
    LocalDocListTrainABr= LocalDocListTrainA+LocalDocListTrainB
    #print("與",keyA, "、",keyB,"個別相關的文章共計",len(LocalDocListTrainABr),"篇")
    if len(LocalDocListTrainAB)!=0:
        KeyAB_EX = expandQuery2(items[1:3], LocalDocListTrainAB)
        EXPANDKEYSlist.append(KeyAB_EX)
        Relationlist.append(relation)
    elif len(LocalDocListTrainABr)!=0:
        KeyABr_EX = expandQuery2(items[1:3],LocalDocListTrainABr)
        EXPANDKEYSlist.append(KeyABr_EX)
        Relationlist.append(relation)
    else:
        EXPANDKEYSlist.append(items[1:3])
        Relationlist.append(relation)
        #print(items[1:3],relation)
print("EXP前剩下",len(corpusListforTrain),"篇文章")

if len(trainList)==len(EXPANDKEYSlist):
    print("MAtch!")    
for key, value in categories.items() :
    print (key, len(value))

explenth=5
while corpusListforTrain!=0 and explenth<20:
    for x in range(len(EXPANDKEYSlist)):
        ExpandKey =EXPANDKEYSlist[x]
        relationbyEXP =Relationlist[x]
        for Docs in corpusListforTrain:
            findkey=0
            for k in ExpandKey[:explenth]:
                if k in Docs:
                    findkey+=1
            if findkey>2:#找到2個以上才算該類
                categories[relationbyEXP].append(Docs)
                #ExpSearchDoc.append(Docs)
                vocabularies[relationbyEXP].extend(ExpandKey[:explenth])
                #vocabularies[relationbyEXP]=list(set(vocabularies[relationbyEXP]))
                corpusListforTrain.pop(corpusListforTrain.index(Docs))
    #print("EXP後剩下",len(corpusListforTrain),"篇文章")
    explenth+=5
print("已分類", len(corpusList)-len(corpusListforTrain),"篇文章", "剩下",len(corpusListforTrain),"篇尚未分類")




#print(ExpSearchDoc)
for key, value in categories.items() :
    print ("categories:",key, len(value))
for key, value in vocabularies.items() :
    print ("vocabularies:",key, len(value))
#處理Vocabulary
for x in vocabularies.keys():
    count = Counter(vocabularies[x]).most_common(100) #按mapping數量排序，取前100個
    vocabularies[x]=[]
    for i in count:
        #print("i ",i)
        vocabularies[x].append(i)
for key, value in vocabularies.items():
    print ("vocabularies new:",key, len(value))

#while corpusListforTrain!=0:
print("分類總數：",len(categories.keys()))
if len(categories.keys())==7:
    relationlist=[]
    for key, value in categories.items():
        relationlist.append(key)
    print(len(relationlist))
    for doc in corpusListforTrain:
        ind = 0
        matchscore = [0,0,0,0,0,0,0]
        matchcounter = 0
        for key in vocabularies:
            #print("KEY= ",key)
            for voc in vocabularies[key]:
                #print("voc= ",voc)
                if voc[0] in doc:
                    matchcounter+=1
            if matchcounter>1:
                matchscore[ind]+=matchcounter/len(voc)
            else:
                matchscore[ind]+=0
            ind+=1
        topRelationAt = numpy.argsort(matchscore)[::-1]
        #print("relationlist[topRelationAt[0]]=",relationlist[topRelationAt[0]])
        categories[relationlist[topRelationAt[0]]].append(doc)
        corpusListforTrain.pop(corpusListforTrain.index(doc))
        #print("matchscore ",matchscore)    
        #print("topRelationAt ",topRelationAt)
        print("剩下",len(corpusListforTrain),"篇未分類")
output = open('Vocabularies.txt', 'ab+')
pickle.dump(vocabularies, output)
output.close()
#while len(corpusListforTrain)!=0: