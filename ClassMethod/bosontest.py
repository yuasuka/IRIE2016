#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 22:47:03 2016

@author: chenyu
"""
import pickle
from functools import reduce
from bosonnlp import BosonNLP
import csv
#import jieba.posseg as pseg
#import jieba as jb
#jb.load_userdict('user_define.txt', )
train = pickle.load(open("wordtype_bosontrain.txt", "rb"))

#nlp = BosonNLP('Zkyyy0Vl.11186.ma9t-5RNnxbh')

#list1 = ["到了二十世纪四十年代，这一问题被美国物理学家理查德费曼、朱利安施温格、日本物理学家朝永振一郎等人突破性地解决了，他们所用的方法被称为重整化。"]
#list2 = ['鬻姒', '齐景公']
#list3 = ['马国凤', '国立中央大学']
'''
for doc in corpus:
    doclist = [item[1] for item in doc]
    if all(x in doclist for x in list1):
        while all(x in doclist for x in list1):
            if all((doclist.index(list1[x+1])-doclist.index(list1[x])==1) for x in range(len(list1)-1)):
                #print("check")
                doclist[doclist.index(list1[0]):doclist.index(list1[0])+len(list1)]=[reduce(lambda x,y: x+y, doclist[doclist.index(list1[0]):doclist.index(list1[0])+len(list1)])]
                #print(doclist[doclist.index(list1[0])],doclist[doclist.index(list1[len(list1)-1])])
                print("doc = ",doclist)

print(len(corpus))
'''
#listall = list1+list2+list3
finallist=[]
finallist1=[item[0] for item in train]
finallist2=[item[1] for item in train]
finallist3=[item[2] for item in train]
with open("output_testwith詞性.csv", 'w',encoding=('utf8')) as f:
    writer1= csv.DictWriter(f, fieldnames = ["A","B","R"])
    writer1.writeheader()
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(finallist1, finallist2,finallist3))