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
import operator


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
    #Keys = jieba.analyse.extract_tags(inputSentence, topK=500, withWeight=False, allowPOS=()) 
    Keys = jieba.cut(inputSentence, cut_all=mode)
    for item in Keys:
        if item not in stopwords and not item.isdigit():
            output.append(item)
    return output

def most_common(lst):
    return max(set(lst), key=lst.count)
    

def expandQuery(queryKey, Dl, weight):
    queryKeyEX =[]
    Dl2= ' '.join(Dl)
    Dllist = Dl2.split(' ')
    checkA=0
    checkB=0
    
    Keys = list(set(Dllist))
    tfarray = numpy.zeros((len(Keys),len(Dl)))
    Cupvarray = numpy.zeros((len(Keys),len(Keys)))
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
    #print("Cuv=\n", Cuv)
    for i in range(len(Keys)):
        for j in range(len(Keys)):
            if Cuv[i][j]!=0:
                Cupvarray[i][j]=Cuv[i][j]/(Cuv[i][i]+Cuv[j][j]-Cuv[i][j])
    #print("CuvP=\n", Cupvarray)
    if checkA!=0:
        Cuv0 = Cupvarray[index0,:]
    else:
        Cuv0 = numpy.zeros((len(Keys),))
    if checkB!=0:
        Cuv1 = Cupvarray[index1,:]
    else:
        Cuv1 = numpy.zeros((len(Keys),))
    Cuvp = numpy.add(Cuv0, Cuv1)
    
    indexlist=numpy.argsort(Cuvp,axis=0)[::-1] #選top N
    #print("與Query:",queryKey,"相關排序",Cuvp,"\n","高度相關key：\n")
    #for x in indexlist:
        #print(Keys[x])
    for i in range(len(indexlist)):
        resultlist = []
        if Cuvp[indexlist[i]]>Cuvp[indexlist[0]]*0 and Keys[indexlist[i]] not in queryKey:
            resultlist.append(Keys[indexlist[i]])
            resultlist.append(Cuvp[indexlist[i]]*weight)
            queryKeyEX.append(resultlist)
    return queryKeyEX    
#####-----Function Area End-----#####





corpus = data_document.split('\n')

typelist = ['n','ng','nr','nrfg','nrt','ns','nt','nz','v','vd','vg','vi','vn','vq']

corpus_selected = corpus[:]
corpusList = []
for i in tqdm(range(len(corpus_selected))):
    corpusKey = Element2Keys(corpus[i], False)
    corpusList.append(corpusKey)

CorpusForTrain = list(corpusList)
trainList.pop(0) #抽掉標題
testList.pop(0)

catlist = ['spouse','parent','child','sibling','birthPlace','deathPlace','workPlace']
categories = {}
categoriesEX = {}
#DocNumberincategories = {}

NotFoundPairList = []
dealDoc = 0
leve1weight = 1

CheckTrainResult = [] #存放有找到的ABpair
others =[]
for item in tqdm(testList):
    keyA = item[1]
    keyB = item[2]
    LocalDocument = []
    LocalDocumentAB = []
    level1worswithweight=[]
    matching = 0
    #-----level 1 trainning-----#
    for corpusD in CorpusForTrain:
        if all(x in corpusD for x in item[1:3]):
            LocalDocument.append(' '.join(corpusD))
            #CorpusForTrain.pop(CorpusForTrain.index(corpusD))
            matching+=1
        else:
            if keyA in corpusD or keyB in corpusD:
                LocalDocumentAB.append(' '.join(corpusD))
                #print("check LocalDocumentAB exist")
                matching+=1
    if matching!=0:
        CheckTrainResult.append(item[1:])
    else:
        others.append(item[1:])

NewOtherList=[]
for name in others:
    newkey01=[]
    newkey02=[]
    KeyEX = jieba.lcut(name[0], cut_all=True)
    for item in KeyEX:
        if item not in stopwords and not item.isdigit():
            newkey01.append(item)
    KeyEX2 = jieba.lcut(name[1], cut_all=True)
    for item2 in KeyEX2:
        if item2 not in stopwords and not item2.isdigit():
            newkey02.append(item2)
    NewOtherList.append([list(newkey01),list(newkey02)])
    #文章一輪找完之後，分析出與AB高度相關的level 1 words，直接進分類，含權重
    #print("find ",keyA, " and ",keyB,"in Docs: ", len(LocalDocument) )
    '''if LocalDocument: #有找到的才做這部分，沒找到就PASS拉
        lv1 = expandQuery(item[1:3], LocalDocument,1)
        categories.setdefault(item[3],[])
        categories[item[3]].extend(lv1)
    if LocalDocumentAB:
        lv2 = expandQuery(item[1:3], LocalDocumentAB,0.5)
        categories.setdefault(item[3],[])
        categories[item[3]].extend(lv2)
        #print("check lv2 exist")'''
print("一共找到多少組？",len(CheckTrainResult))
#for k, v in categories.items():
    #print(k,v)
#--------------------Check the train model--------------------#
'''#2016 12 17 night note:可能要把權重也要算進去，也許答案會很不一樣
scorelist = {}
#scorelistDoc={}
score = 0
for i in range(7):
    scorelist.setdefault(catlist[i],0)
    #scorelistDoc.setdefault(catlist[i],0)
CorpusForCheck = list(corpusList)
##----------方式3 Doc分離後，計算相關性高的words x vocabulary----------##
score=0
for x in tqdm(trainList[:5]):
    LocalDocForCheck=[]
    LocalDocForEXCheck=[]
    for i in range(7):
        scorelist[catlist[i]]=0
    for Doc in CorpusForCheck:
        if x[1] in Doc and x[2] in Doc:
            LocalDocForCheck.append(' '.join(Doc))
            CorpusForCheck.pop(CorpusForCheck.index(Doc))
        else:
            if x[1] in Doc or x[2] in Doc:
                LocalDocForCheck.append(' '.join(Doc))
    CheckAnalysisWords= expandQuery(x[1:3],LocalDocForCheck,1)    
    CheckAnalysisWords2= expandQuery(x[0:2],LocalDocForEXCheck,0.5)
    #print("check point 1, round ", CheckTrainResult.index(x))
    CheckAnalysisWords=CheckAnalysisWords.extend(CheckAnalysisWords2)
    for analywords in CheckAnalysisWords:
        for k,v in categories.items():
            for key in v:
                if analywords[0] ==key[0]:
                    scorelist[k]+=key[1]*analywords[1]
    #print("check point 2, round ", CheckTrainResult.index(x))
    for analywords2 in CheckAnalysisWords2:
        for k2, v2 in categoriesEX.items():
            for key2 in v2:
                if analywords2[0] ==key2[0]:
                    scorelist[k2]+=key2[1]*analywords2[1]*0.5
    EstimateResult =max(scorelist, key=(lambda key:scorelist[key]))
    #print("check point 3, round ", CheckTrainResult.index(x))
    if EstimateResult==x[3]:
        score+=1
    else:
        pass
        print("scorelist",scorelist)
        print("錯誤囉！", x[1:3],"推估的類型為：",EstimateResult,"但正確應為：",x[3])
print("Training Check方式3 結果，準確率為：", score*100/len(CheckTrainResult),"%")
#--------------------Test Start--------------------#
idList=[]
n=1
for item in testList:
    idList.append(item[0])  
CorpusForTest = list(corpusList)
answerList=[]
counter=0
for x in tqdm(testList):
    LocalDocForCheck=[]
    find= 0
    for i in range(7):
        scorelist[catlist[i]]=0
    for Doc in CorpusForTest:
        if x[1] in Doc and x[2] in Doc:
            LocalDocForCheck.append(' '.join(Doc))
            find+=1
    if find==0:
        counter+=1
    CheckAnalysisWords= expandQuery(x[0:2],LocalDocForCheck)
    for analywords in CheckAnalysisWords:
        for k,v in categories.items():
            for key in v:
                if analywords[0] ==key[0]:
                    scorelist[k]+=key[1]*analywords[1]
        for k2, v2 in categoriesEX.items():
            for key2 in v2:
                if analywords[0] ==key2[0]:
                    scorelist[k2]+=key2[1]*analywords[1]*0.5
    EstimateResult =max(scorelist, key=(lambda key:scorelist[key]))
    answerList.append(EstimateResult)
print("有多少對沒找到成對的：",counter)


with open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/TermProj02_1219a.csv", 'w') as f:
    writer1= csv.DictWriter(f, fieldnames = ["Id", "Property"])
    writer1.writeheader()
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(idList, answerList))'''