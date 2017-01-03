# -*- coding: utf-8 -*-

import jieba
import csv
import pickle
from tqdm import * 

jieba.load_userdict('user_define.txt')

data = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/ref_text.txt", 'r', encoding=('utf8')).read()
stopwords = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/stopwords.txt", 'r', encoding=('utf8')).read()

corpus = data.split('\n')

CORPUS=[]
for doc in tqdm(corpus[:]):
    temp = jieba.lcut(doc, cut_all=False)
    Docs=[]
    for s in temp:
        if s not in stopwords and not s.isdigit():
            Docs.append(s)
    CORPUS.append(Docs)
#print(CORPUS)

with open("Corpus1225.txt", "wb") as fp:
    pickle.dump(CORPUS, fp)