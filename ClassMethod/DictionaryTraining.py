# -*- coding: utf-8 -*-

import numpy
import csv
import pickle
from tqdm import tqdm
from bosonnlp import BosonNLP
#nlp = BosonNLP('Zkyyy0Vl.11186.ma9t-5RNnxbh')

#File loading
stopwords = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/stopwords.txt", 'r', encoding=('utf8')).read()
#data_document = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/ref_text.txt", 'r', encoding=('utf8')).read()
trainList = list(csv.reader(open('train.csv', 'r', encoding=('utf8'))))
testList = list(csv.reader(open('test.csv', 'r', encoding=('utf8'))))
corpus = pickle.load(open("Corpus_wordtype_bosontrain1222c.txt","rb"))
train = pickle.load(open("wordtype_bosontrain.txt", "rb"))

corpuslistAandB = pickle.load(open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/JiebaFiles/CorpusAandBFound_J.txt", "rb"))
corpuslistAorB = pickle.load(open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/JiebaFiles/CorpusAorBFound_J.txt", "rb"))
TrainlistAandB = pickle.load(open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/JiebaFiles/TrainABFound_J.txt", "rb"))
TrainlistAorB = pickle.load(open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/JiebaFiles/TrainAorBFound_J.txt", "rb"))




#####-----Function Area-----#######  
def functionbox(doc,words):
    dontwantlist=['y','o','h','p','pba','pbei','a','vshi','vyou','c','f',
    'm','q','wkz','wj','wd','wky','wyz','wyy','ww','wt','wf','wn','wm',
    'ws','wp','wb','wh','uzhe','ule','uguo','ude','usuo','udeng','uyy',
    'udh','uzhi','ulian']
    for item in doc:
        #這邊可以注意一下，使用and 跟or效果不一樣
        if (item[1] in words) or (item[0] not in dontwantlist):
            NewDoc2.append(item[1])
    return NewDoc2

def foundbox(doc,pair,pair0,pair1,pair2):
    #print(pair0,pair1,pair2)
    NewDoc2 = []
    words=[]
    for y in pair[1:3]:
        words.extend(y)
    #print("Word=",words)
    #print("Pair",pair)
    if any(x in [item[1] for item in doc] for x in words):            
            if all(x in [item[1] for item in doc] for x in words): #AB即使雜亂也都在裡面，但因為有被打斷的，不能肯定一定有mapping到
                if any(len(x)>1 for x in pair[1:3]):
                    if len(pair[1])>1 and len(pair[2])==1:      #A被切了
                        if (all(x in [item[1] for item in doc] for x in pair[1])):
                            if all(([item[1] for item in doc].index(pair[1][x+1])-[item[1] \
                            for item in doc].index(pair[1][x])==1) for x in range(len(pair[1])-1)):
                                pair2=1
                    elif len(pair[1])==1 and len(pair[2])>1:    #B被切了
                        if (all(x in [item[1] for item in doc] for x in pair[2])):
                            if all(([item[1] for item in doc].index(pair[2][x+1])-[item[1] \
                            for item in doc].index(pair[2][x])==1) for x in range(len(pair[2])-1)):
                                pair2=1   
                    else: #兩個都斷了
                        if all(([item[1] for item in doc].index(pair[1][x+1])-[item[1] for item in doc].index(pair[1][x])==1) for x in range(len(pair[1])-1)) \
                        and all(([item[1] for item in doc].index(pair[2][x+1])-[item[1] for item in doc].index(pair[2][x])==1) for x in range(len(pair[2])-1)):
                            pair2=1  
                else:#AB都不斷
                    pair2=1  
            else:#AB只有一個有出現的
                pair1=1
    else: #AB半個都沒出現的
            #right=4
            pair0=1
    if pair2!=0:
        NewDoc2 = functionbox(doc,words)
        #localcorpusAandB.append(NewDoc2)
    #print("NewDoc2",NewDoc2,"pair2",pair2,"pair1",pair1,"pair0",pair0)
    return NewDoc2,pair1,pair1,pair2
def expandQuery(queryKey, Dl, weight):
    queryKeyEX =[]
    Dl2=[]
    for doc in Dl:
        Dl2.extend(doc)
    #print("query",queryKey)
        #print(len(Dl),Dl2)
    #Dllist = Dl2.split(' ')
    #print("Dl2: ",Dl2)
    Keys = list(set(Dl2))
    tfarray = numpy.zeros((len(Keys),len(Dl)))
    Cuvparray = numpy.zeros((len(Keys),len(Keys)))
    indexOfQuery = []
    for i in range(len(Keys)):
        if Keys[i] in queryKey:
            indexOfQuery.append(i)
        for j in range(len(Dl)):
            if Keys[i] in Dl[j]:
                tfarray[i][j]+=1
            else:
                tfarray[i][j]+=0
    #print("indexOfQuery",list(set(indexOfQuery)))
    Cuv = tfarray.dot(tfarray.T)
    #print("Cuv=\n", Cuv)
    for i in range(len(Keys)):
        for j in range(len(Keys)):
            if Cuv[i][j]!=0:
                Cuvparray[i][j]=Cuv[i][j]/(Cuv[i][i]+Cuv[j][j]-Cuv[i][j])
    #print("CuvP=\n", Cuvparray)
    #Cuv0 = numpy.zeros((len(Keys),))
    #for x in indexOfQuery:
    #    if indexOfQuery.index(x)==0:
    #        Cuvp = numpy.add(Cuv0,Cuvparray[x,:])
     #   else:
     #       Cuvp = numpy.add(Cuvp,Cuvparray[x,:])
    
    #print("Cuvp", Cuvp)
    #planB
    Cuvp=numpy.zeros((len(Keys),))
    for i in indexOfQuery:
        if indexOfQuery.index(i)==0:
            #print("in ",i)
            cuvp = Cuvparray[i,:]
        else:
            #print("in ",i)
            Cuvp1 = Cuvparray[i,:]
            for x in range(len(Keys)):
                temp = cuvp[x]*Cuvp1[x]
                Cuvp[x]+=temp
    #print(Cuvp)
    indexlist2=numpy.argsort(Cuvp,axis=0)[::-1] #選top N
    for i in range(len(indexlist2)):
        resultlist = []
        if Cuvp[indexlist2[i]]>0 and Keys[indexlist2[i]] not in queryKey:
            resultlist.append(Keys[indexlist2[i]])
            resultlist.append(Cuvp[indexlist2[i]]/Cuvp[indexlist2[0]])
            queryKeyEX.append(resultlist)
    return queryKeyEX
#####-----------------------Function Area End-----------------------#####
Vocabularies={}
corpusfortrain = list(corpuslistAandB)
trainlist=[]
for pair in train[200:220]:
    I= train.index(pair)+1
    A= pair[0]
    B= pair[1]
    R= pair[2]
    #trainlist已經不包含詞性了，只留下文字，後續要比對的話要用原始的train
    trainlist.append([I,A[int(len(A)*0.5):],B[int(len(B)*0.5):],R])        
    #print(int(len(pair)*0.5))
#print(trainlist)
#想排除的詞性
findPairs1=0
findPairs2=0
allpairfound=[] #AB都找到的放在這邊
notallpairfound=[] #沒有AB都找到的放在這邊
allpairnotfound=[] #AB都找不到的放在這邊
corpusSelect = list(corpus[:])
for pair in tqdm(trainlist[:], desc="Training start"):
    #print("pair", pair)
    localcorpusAandB = []
    localcorpusAorB = []
    templocalCorpus = []
    for doc in corpusSelect:
        NewDoc0 = []               
        NewDoc2 = []
        pair2 =0
        pair1 =0
        pair0 =0
        NewDoc2,pair0,pair1,pair2=foundbox(doc, pair, pair0,pair1,pair2)
        if NewDoc2:
            localcorpusAandB.append(NewDoc2)
    #print("localcorpusAandB",len(localcorpusAandB))
    #if localcorpusAandB:
        #allpairfound.append(pair)
    #elif pair2==0 and pair1!=0:
        #notallpairfound.append(pair)
    #elif pair2==pair1==0 and pair0!=0:
        #allpairnotfound.append(pair)
    if localcorpusAandB:
        #print(pair)
        findPairs2+=1
        allpairfound.append(pair)
        realtionWords = expandQuery([item for sublist in pair[1:3] for item in sublist], localcorpusAandB,1)
        Vocabularies.setdefault(pair[3],[])
        Vocabularies[pair[3]].extend(realtionWords)
    '''if localcorpusAorB:
        #print(pair)
        findPairs1+=1
        realtionWords = expandQuery(words, localcorpusAorB,0.5)
        Vocabularies.setdefault(pair[2],[])
        Vocabularies[pair[2]].extend(realtionWords)'''
#print("找到的雙Pair數: ",findPairs2)
#print("找到的單Pair數: ",findPairs1)'''
typebaseline={}
sumthenumber=0
for k, v in Vocabularies.items():
    sumthenumber=sum(len(v) for v in Vocabularies.values())
    typebaseline.setdefault(k,[])
    typebaseline[k]=round(len(Vocabularies[k])/sumthenumber,4)
for k, v in typebaseline.items():
    typebaseline[k]=round(v*100/sumthenumber,2)
    print(k, v,"%")
print("sum =", sumthenumber)
#-------------------Vocabularies 1 ready----------------------#
print("Start to check train")
score=0
scorelist ={}
BSscorelist ={}
catlist=[]
localcorpusTAandB=[]
for k,v in Vocabularies.items():
    catlist.append(k)
    scorelist.setdefault(k,0)
    BSscorelist.setdefault(k,0)
trainlistforcheck=allpairfound[:]
for t in tqdm(trainlistforcheck[:], desc="Time to check the model"):
    LocalDocForCheck=[]
    pair2 =0
    pair1 =0
    pair0 =0
    #LocalDocForEXCheck=[]
    wordsT=[]
    for y in t[1:3]:
        wordsT.extend(y)
    
    for i in range(len(catlist)):
        scorelist[catlist[i]]=0
        BSscorelist[catlist[i]] =0
    #print("wordsT")       
    for Doc in corpusSelect:
        pair2 =0
        pair1 =0
        pair0 =0
        NewDoc3,pair0,pair1,pair2=foundbox(Doc, t, pair0,pair1,pair2)
        if NewDoc3:
            LocalDocForCheck.append(NewDoc3)
    #print("check point 1, round ")
    if LocalDocForCheck:
        #print("get in  LocalDocForCheck, lenth=",len(LocalDocForCheck))
        CheckAnalysisWords= expandQuery(wordsT,LocalDocForCheck,1)    
        #print("check point 2, round ")
        #--------------------詞性檢查--------------------#
        condition=[item[0] for item in train[t[0]-1]][0:2]
        result=[]
        if all((p=='nr'or p=='nrf' or p=='nz') for p in condition):
            #scorelist['birthPlace']=scorelist['deathPlace']=scorelist['workPlace']=0
            #print("rightB")
            for words in CheckAnalysisWords:
                for k,v in Vocabularies.items():
                    if k in ['child','parent','spouse','sibling']:
                        for termkey in Vocabularies[k]:
                            if words[0] ==termkey[0]:
                                scorelist[k]+=termkey[1]*words[1]
            for i,j in scorelist.items():
                if i in typebaseline.keys():
                    BSscorelist[i]=scorelist[i]*typebaseline[i]
        else:
            #scorelist['child']=scorelist['parent']=scorelist['spouse']=scorelist['sibling']=0
            #print("rightP")
            for words in CheckAnalysisWords:
                for k,v in Vocabularies.items():
                    if k in ['birthPlace','deathPlace','workPlace']:
                        for termkey in Vocabularies[k]:
                            if words[0] ==termkey[0]:
                                scorelist[k]+=termkey[1]*words[1]
            for i,j in scorelist.items():
                if i in typebaseline.keys():
                    BSscorelist[i]=scorelist[i]/typebaseline[i]
        EstimateResult =max(scorelist, key=(lambda key:scorelist[key]))
        if EstimateResult==t[3]:
            score+=1
        else:
            pass
            print("scorelist",scorelist)
            print("錯誤囉！", t[1:3],"推估的類型為：",EstimateResult,"但正確應為：",t[3])
    else:
        print("在corpus中沒有找到喔")
print("Training Check方式3 結果，準確率為：", score*100/len(trainlistforcheck),"%")

'''#--------------------Check the train model--------------------#
#2016 12 17 night note:可能要把權重也要算進去，也許答案會很不一樣
scorelist = {}
#scorelistDoc={}
score = 0
for i in range(7):
    scorelist.setdefault(catlist[i],0)
    #scorelistDoc.setdefault(catlist[i],0)
CorpusForCheck = list(corpusList)
##----------方式3 Doc分離後，計算相關性高的words x vocabulary----------##

output = open('Vocabularies.txt', 'ab+')
pickle.dump(vocabularies, output)
output.close()'''