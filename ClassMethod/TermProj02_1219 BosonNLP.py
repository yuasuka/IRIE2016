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
import time
nlp = BosonNLP('Zkyyy0Vl.11186.ma9t-5RNnxbh')

jieba.load_userdict('user_define.txt', )
jieba.analyse.set_stop_words('stopwords2.txt')

#File loading--Mac
#stopwords = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/stopwords.txt", 'r', encoding=('utf8')).read()
#data_document = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/ref_text.txt", 'r', encoding=('utf8')).read()
#trainList = list(csv.reader(open('train.csv', 'r', encoding=('utf8'))))
#testList = list(csv.reader(open('test.csv', 'r', encoding=('utf8'))))
#File loading--Win
stopwords = open("stopwords.txt", 'r', encoding=('utf8')).read()
data_document = open("ref_text.txt", 'r', encoding=('utf8')).read()
trainList = list(csv.reader(open('train.csv', 'r', encoding=('utf8'))))
testList = list(csv.reader(open('test.csv', 'r', encoding=('utf8'))))

corpus = data_document.split('\n')

#------------------Corpus------------------#
corpus_selected = corpus[:]
corpusList = []
for x in tqdm(corpus_selected[1:2]):
    #corpusKey = jieba.lcut(corpus[i], cut_all=False) #nlp.tag(corpus[i], oov_level = 4)
    TrainKey = nlp.tag(x[1:3], oov_level=4)
    time.sleep(0.5)
    temp=[]
    for words in TrainKey:
        if words['word'] not in stopwords:
            temp.extend([words['tag']+words['word']])
    temp.extend([x[3]])
    corpusList.append(temp)
    print(corpusList)
    #Element2Keys(corpus[i], False)

trainListnew=[]
'''for x in tqdm(trainList[1:]):
    #corpusKey = jieba.lcut(corpus[i], cut_all=False) #nlp.tag(corpus[i], oov_level = 4)
    TrainKey = nlp.tag(x[1:3], oov_level=4)
    time.sleep(0.5)
    temp=[]
    for words in TrainKey:
        temp.extend([words['tag']+words['word']])
    temp.extend([x[3]])
    trainListnew.append(temp)
    #print(trainListnew)
    #Element2Keys(corpus[i], False)'''
with open("wordtype_bosontrain.txt", "wb") as fp:
    pickle.dump(trainListnew, fp)

testListnew=[]
'''for x in tqdm(testList[1:]):
    TrainKey = nlp.tag(x[1:3], oov_level=4)
    time.sleep(0.5)
    temp2=[]
    for words in TrainKey:
        temp2.extend([words['tag']+words['word']])
    testListnew.append(temp2)
with open("wordtype_bosontest.txt", "wb") as tp:
    pickle.dump(testListnew, tp)'''

'''
global CorpusQueue
CorpusQueue=Queue()
global NewCorpusQueue
NewCorpusQueue=Queue()
global bar

def workers():
    while not CorpusQueue.empty():
        corpus =[]
        corpus=CorpusQueue.get()
        corpusKey = nlp.tag(corpus, oov_level=4)
        time.sleep(0.6)
        temp=[]
        for words in corpusKey[0]['word']:
            if words not in stopwords and not words.isdigit():
                temp.append(words)
        NewCorpusQueue.put(temp)
        bar.update()
    return
    
for c in corpus_selected:
    CorpusQueue.put(c)
length = len(corpus_selected)
    
bar = pyprind.ProgPercent(length, track_time=True)    
t={}
for x in range(10):
    t['threads{}'.format(x)]=Thread(target=workers)
for p in t.keys():
    t[p].start()
for p in t.keys():
    t[p].join()

for c2 in range(NewCorpusQueue.qsize()):
    cnew=NewCorpusQueue.get()
    corpusList.append(cnew)'''


        
