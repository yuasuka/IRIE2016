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
jieba.analyse.set_stop_words('stopwords2.txt')

#File loading
stopwords = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/stopwords.txt", 'r', encoding=('utf8')).read()
data_document = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/ref_text.txt", 'r', encoding=('utf8')).read()
trainList = list(csv.reader(open('train.csv', 'r', encoding=('utf8'))))
testList = list(csv.reader(open('test.csv', 'r', encoding=('utf8'))))



#####-----Function Area-----#######
def Element2Keys(inputSentence, mode):
    output = []
    Keys = jieba.analyse.extract_tags(inputSentence, topK=500, withWeight=False, allowPOS=()) #jieba.cut(inputSentence, cut_all=mode)
    for item in Keys:
        if item not in stopwords and not item.isdigit():
            output.append(item)
    return output

def most_common(lst):
    return max(set(lst), key=lst.count)
    

def expandQuery(queryKey, Dl,expandnumber):
    queryKeyEX =[]
    verctornizer = TfidfVectorizer()
    tfidf = verctornizer.fit_transform(Dl)
    RelativeScore = tfidf.T.dot(tfidf)
    words = verctornizer.get_feature_names()
    submatrix = RelativeScore[:,words.index(queryKey)]
    array = submatrix.toarray()
    #print(array)
    indexlist=numpy.argsort(array,axis=0)[::-1][:expandnumber] #選top N
    #print(indexlist)
    for i in range(len(indexlist)):
        queryKeyEX.append(words[indexlist[i]])
    #print(queryKeyEX)
    return queryKeyEX
#####-----Function Area End-----#####





corpus = data_document.split('\n')

typelist = ['n','ng','nr','nrfg','nrt','ns','nt','nz','v','vd','vg','vi','vn','vq']

corpus_selected = corpus[:10000]
corpusList = []
for i in tqdm(range(len(corpus_selected))):
    corpusKey = Element2Keys(corpus[i], False)
    corpusList.append(corpusKey)

trainList.pop(0) #抽掉標題
testList.pop(0)

catlist = ['spouse','parent','child','sibling','birthPlace','deathPlace','workPlace']
categories = {}
NotFoundPairList = []
dealDoc = 0
for item in tqdm(trainList[:1000]):
    keyA = item[1]
    keyB = item[2]
    categories.setdefault(item[3],[])    #print("item = ", item)
    LocalDocument = []
    findPairs = 0
    for corpusD in corpusList:
        if (keyA in corpusD) and (keyB in corpusD):
            #print("Find A&B")
            categories[item[3]].append(corpusD) #在某一個分類中，把Doc丟進去
            corpusList.pop(corpusList.index(corpusD))
            findPairs=1
    if findPairs == 0:
        NotFoundPairList.append(item) #找不到的train成一個新的list之後expand再繼續找
    
#Expansion for query
indlist = []
findPairsByEXQ=0
#while len(corpusList)-len(indlist)>0:
expandnumber = 5
catedDoc = 0
print("corpuslist before expansion", len(corpusList))
for items in tqdm(NotFoundPairList):
    keyAN = items[1]
    keyBN = items[2]
    relation = items[3]
    LocalDocListA = []
    LocalDocListB = []
    for Doc in corpusList:
        if keyAN in Doc:
            LocalDocListA.append(' '.join(Doc))
        elif keyBN in Doc:
            LocalDocListB.append(' '.join(Doc))
        else:
            pass
    if len(LocalDocListA)!=0 and len(LocalDocListB)!=0:
        #print("find in AEX and BEX")
        KeyAN_EX = expandQuery(keyAN,LocalDocListA,expandnumber)
        KeyBN_EX = expandQuery(keyBN,LocalDocListB,expandnumber)
        #print("KeyA EX",'\n',KeyAN_EX)
        #print("KeyB EX",'\n',KeyBN_EX)
        LocalDocFromEXA=[]
        LocalDocFromEXB=[]
        findA = 0
        findB = 0
        for Doc in corpusList:
            for i in range(min(len(KeyAN_EX),len(KeyBN_EX))):
                if KeyAN_EX[i] in Doc:
                    findA+=1
                if KeyBN_EX[i] in Doc:
                    findB+=1
            if findA!=0 and findB!=0:
                categories[relation].append(Doc)
                corpusList.pop(corpusList.index(Doc))
                catedDoc+=1
    elif len(LocalDocListA)!=0 and len(LocalDocListB)==0:
        #KeyAN_EX = expandQuery(keyAN,LocalDocListA)
        for Doc in corpusList:
            if keyAN in Doc:
                categories[relation].append(Doc)
                corpusList.pop(corpusList.index(Doc))
                catedDoc+=1
            else:
                pass
        #print(LocalDocFromEXA)
    elif len(LocalDocListA)==0 and len(LocalDocListB)!=0:
        #KeyBN_EX = expandQuery(keyBN,LocalDocListB)
        for Doc in corpusList:
            if keyBN in Doc:
                categories[relation].append(Doc)
                corpusList.pop(corpusList.index(Doc))
                catedDoc+=1
            else:
                pass
        #print(LocalDocFromEXB)
    else:
        #print("Two Key not found in any docs")
        pass
print(len(corpusList), "Done Doc= ", catedDoc)
        