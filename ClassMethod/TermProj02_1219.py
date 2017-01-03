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
    

def expandQuery(queryKey, Dl):
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
            resultlist.append(Cuvp[indexlist[i]])
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
for item in tqdm(trainList[:500]):
    keyA = item[1]
    keyB = item[2]
    LocalDocument = []
    LocalDocumentAB = []
    level1worswithweight=[]
    matching = 0
    #-----level 1 trainning-----#
    for corpusD in CorpusForTrain:
        if (keyA in corpusD) or (keyB in corpusD):
            LocalDocument.append(' '.join(corpusD))  #AB都出現的level 1文章
            #print("Find A&B")
            #categories[item[3]].append(corpusD) #在某一個分類中，把Doc丟進去
            #CorpusForTrain.pop(CorpusForTrain.index(corpusD))
            matching+=1
        #-----level 2 trainning-----#
        '''elif (keyA in corpusD) and (keyB not in corpusD):
            LocalDocumentAB.append(' '.join(corpusD))
            matching+=1
        elif (keyA not in corpusD) and (keyB in corpusD):
            LocalDocumentAB.append(' '.join(corpusD))
            matching+=1'''
    if matching!=0:
        CheckTrainResult.append(item[1:])
    #文章一輪找完之後，分析出與AB高度相關的level 1 words，直接進分類，含權重
    #print("find ",keyA, " and ",keyB,"in Docs: ", len(LocalDocument) )
    if LocalDocument: #有找到的才做這部分，沒找到就PASS拉
        lv1 = expandQuery(item[1:3], LocalDocument)
        #lv2 = expandQuery(item[1:3], LocalDocumentAB)
        if lv1 or lv2:
            #level1worswithweight.append(lv1)
            #level1worswithweight.append(str(leve1weight))
            categories.setdefault(item[3],[])    #print("item = ", item)
            #categoriesEX.setdefault(item[3],[])    #print("item = ", item)
            #DocNumberincategories.setdefault(item[3],0)    #print("item = ", item)
            categories[item[3]].extend(lv1)
            #categoriesEX[item[3]].extend(lv2)
            #DocNumberincategories[item[3]]+=1    
            #print("item = ", item)
print("一共找到多少組？",len(CheckTrainResult))
#for k, v in categories.items():
    #print(k,v)
#--------------------Check the train model--------------------#
#2016 12 17 night note:可能要把權重也要算進去，也許答案會很不一樣
scorelist = {}
#scorelistDoc={}
score = 0
for i in range(7):
    scorelist.setdefault(catlist[i],0)
    #scorelistDoc.setdefault(catlist[i],0)
CorpusForCheck = list(corpusList)
##----------方式1 Doc中尋找到vocabulary中的字，把Doc數加起來----------##
'''for x in tqdm(CheckTrainResult):
    for i in range(7):
        scorelist[catlist[i]]=0
        scorelistDoc[catlist[i]]=0
    for Doc in CorpusForCheck:
        FindinDoc=0
        if x[0] in Doc and x[1] in Doc:
            for words in Doc:
                for k,v in categories.items():
                    for key in v:
                        if words ==key[0]:
                            FindinDoc+=1
                        else:
                            FindinDoc+=0
                    scorelist[k]+=FindinDoc
            EstimateResultDoc =max(scorelist, key=(lambda key:scorelist[key]))
            scorelistDoc[EstimateResultDoc]+=1
    #print("scorelist",scorelist)
    for k,v  in scorelistDoc.items():
        if DocNumberincategories[k]:
            scorelistDoc[k]=scorelistDoc[k]/DocNumberincategories[k]
    EstimateResult =max(scorelistDoc, key=(lambda key:scorelistDoc[key]))
    #print("EstimateResult",EstimateResult)
    if EstimateResult==x[2]:
        score+=1
    else:
        pass
        #print("scorelist",scorelist)
        #print("錯誤囉！", x[0:2],"推估的類型為：",EstimateResult,"但正確應為：",x[2])
print("Training Check方式1 結果，準確率為：", score*100/len(CheckTrainResult),"%")'''

##----------方式2 Doc中尋找到vocabulary中的字，把權重加起來----------##
'''score=0
for x in tqdm(CheckTrainResult):
    for i in range(7):
        scorelist[catlist[i]]=0
    for Doc in CorpusForCheck:
        if x[0] in Doc and x[1] in Doc:
            for words in Doc:
                for k,v in categories.items():
                    for key in v:
                        if words ==key[0]:
                            scorelist[k]+=key[1]
                for k2,v2 in categoriesEX.items():
                    for key2 in v2:
                        if words ==key2[0]:
                            scorelist[k2]+=key2[1]*0.5 #關係比較遠一點
        #print("scorelist",scorelist)
    EstimateResult =max(scorelist, key=(lambda key:scorelist[key]))
    #print("EstimateResult",EstimateResult)
    if EstimateResult==x[2]:
        score+=1
    else:
        pass
        #print("scorelist",scorelist)
        #print("錯誤囉！", x[0:2],"推估的類型為：",EstimateResult,"但正確應為：",x[2])
print("Training Check方式2 結果，準確率為：", score*100/len(CheckTrainResult),"%")'''
##----------方式3 Doc分離後，計算相關性高的words x vocabulary----------##
score=0
for x in tqdm(CheckTrainResult[:500]):
    LocalDocForCheck=[]
    LocalDocForEXCheck=[]
    for i in range(7):
        scorelist[catlist[i]]=0
    for Doc in CorpusForCheck:
        if x[0] in Doc or x[1] in Doc:
            LocalDocForCheck.append(' '.join(Doc))
        #elif ((x[0] in Doc) and (x[1] not in Doc)) and ((x[0] not in Doc) and (x[1] in Doc)):
            #LocalDocForEXCheck.append(' '.join(Doc))
    CheckAnalysisWords= expandQuery(x[0:2],LocalDocForCheck)    
    #CheckAnalysisWords2= expandQuery(x[0:2],LocalDocForEXCheck)
    for analywords in CheckAnalysisWords:
        for k,v in categories.items():
            for key in v:
                if analywords[0] ==key[0]:
                    scorelist[k]+=key[1]*analywords[1]
    '''for analywords2 in CheckAnalysisWords2:
        for k2, v2 in categoriesEX.items():
            for key2 in v2:
                if analywords2[0] ==key2[0]:
                    scorelist[k2]+=key2[1]*analywords2[1]*0.5'''
    EstimateResult =max(scorelist, key=(lambda key:scorelist[key]))
    if EstimateResult==x[2]:
        score+=1
print("Training Check方式3 結果，準確率為：", score*100/len(CheckTrainResult),"%")
#--------------------Test Start--------------------#
'''idList=[]
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