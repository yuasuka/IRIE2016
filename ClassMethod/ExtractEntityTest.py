# -*- coding: utf-8 -*-

import jieba
import csv
import pickle
from tqdm import * 
import jieba.posseg as pseg
import math

'''
jieba.load_userdict('user_define.txt')

data = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/ref_text.txt", 'r', encoding=('utf8')).read()
stopwords = open("/Users/chenyu/Dropbox/IR2016F/ir_project2_data/stopwords.txt", 'r', encoding=('utf8')).read()

corpus = data.split('\n')
print(corpus[1])
words = pseg.cut(corpus[1])

for x in words:
	print(list(x)[0]+"/"+list(x)[1], end=" ")'''

G1 = [1, 0, 1, 0, 0, 3, 0 , 0 , 0, 2, 0 , 0 , 0 , 0, 3]
CG1= []
s = 0
for x in G1:
	s = s +x
	CG1.append(s)
DCG= []
for i in range(len(G1)):
	if i==0:
		s = G1[i]
	else:
		s = DCG[i-1]+G1[i]/(math.log2(i+1))
	DCG.append(s)
print("CG1 = ", CG1)
print("DCG1 = ", DCG)