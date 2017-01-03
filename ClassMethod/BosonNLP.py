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
from bosonnlp import BosonNLP
import pickle
from queue import Queue
from threading import Thread
import pyprind
import time
nlp = BosonNLP('Zkyyy0Vl.11186.ma9t-5RNnxbh')

jieba.load_userdict('user_define.txt', )
jieba.analyse.set_stop_words('stopwords2.txt')

#File loading
stopwords = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/stopwords.txt", 'r', encoding=('utf8')).read()
data_document = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/ref_text.txt", 'r', encoding=('utf8')).read()
trainList = list(csv.reader(open('train.csv', 'r', encoding=('utf8'))))
testList = list(csv.reader(open('test.csv', 'r', encoding=('utf8'))))

corpus = data_document.split('\n')

corpus_selected = corpus[:]
corpusList = []
trainListnew=[]
#------------------------------Corpus------------------------------#
'''for x in tqdm(corpus_selected[:1]):
    #corpusKey = jieba.lcut(corpus[i], cut_all=False) #nlp.tag(corpus[i], oov_level = 4)
    TrainKey = nlp.tag(x, oov_level=4)
    time.sleep(0.5)
    #print(len(TrainKey))
    temp=[]
    for words in TrainKey:
       # print(words,'\n')
        #if words['word'] not in stopwords:
        for x in range(len(words['word'])):
            if words['tag'][x] not in ['t','wj','wd','u']:
                temp.append([words['tag'][x],words['word'][x]])
corpusList.append(temp)'''
#print(corpusList)
    #Element2Keys(corpus[i], False)
#with open("Corpus_wordtype_bosontrain.txt", "wb") as fp:
#    pickle.dump(corpusList, fp)

'''
for x in tqdm(trainList[1:]):
    #corpusKey = jieba.lcut(corpus[i], cut_all=False) #nlp.tag(corpus[i], oov_level = 4)
    TrainKey = nlp.tag(x[1:3], oov_level=4)
    time.sleep(0.5)
    temp=[]
    for words in TrainKey:
        temp.extend([words['tag']+words['word']])
    temp.extend([x[3]])
    trainListnew.append(temp)
    #print(trainListnew)
    #Element2Keys(corpus[i], False)
with open("wordtype_bosontrain.txt", "wb") as fp:
    pickle.dump(trainListnew, fp)

testListnew=[]
for x in tqdm(testList[1:]):
    TrainKey = nlp.tag(x[1:3], oov_level=4)
    time.sleep(0.5)
    temp2=[]
    for words in TrainKey:
        temp2.extend([words['tag']+words['word']])
    testListnew.append(temp2)
with open("wordtype_bosontest.txt", "wb") as tp:
    pickle.dump(testListnew, tp)'''


global CorpusQueue
CorpusQueue=Queue()
global NewCorpusQueue
NewCorpusQueue=Queue()
global bar

def workers():
    while not CorpusQueue.empty():
        corpus =[]
        corpus=CorpusQueue.get()
        TrainKey = nlp.tag(corpus, oov_level=4)
        time.sleep(0.5)
        #print(len(TrainKey))
        temp=[]
        for words in TrainKey:
        # print(words,'\n')
            #if words['word'] not in stopwords:
            for x in range(len(words['word'])):
                #if words['tag'][x] not in ['wkz','wj','wd','wky','wyz','wyy','ww','wt','wf','wn','wm','ws','wp','wb','wh','uzhe','ule','uguo','ude','usuo','udeng','uyy','udh','uzhi','ulian']:
                temp.append([words['tag'][x],words['word'][x]])
        NewCorpusQueue.put(temp)
        #print(CorpusQueue.qsize())
        bar.update()
    return 
    
for c in corpus_selected:
    CorpusQueue.put(c)
length = len(corpus_selected)
    
bar = pyprind.ProgPercent(length, track_time=True)    
t={}
for x in range(20):
    t['threads{}'.format(x)]=Thread(target=workers)
for p in t.keys():
    t[p].start()
for p in t.keys():
    t[p].join()

counter=0
#if NewCorpusQueue.qsize()==length:
#    print("get in check!")
while not NewCorpusQueue.empty():
    cnew=NewCorpusQueue.get()
    corpusList.append(cnew)
    counter+=1

#print(corpusList)
with open("Corpus_wordtype_bosontrain1222c.txt", "wb") as fp:
    pickle.dump(corpusList, fp)