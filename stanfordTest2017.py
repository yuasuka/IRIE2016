from __future__ import unicode_literals
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# coding: utf8
"""
Created on Mon Jan  2 11:26:39 2017

@author: yuchen
"""
import pickle
import nltk
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser

import os
os.environ['CLASSPATH'] ="SF2017/stanford-segmenter/stanford-segmenter.jar:\
SF2017/stanford-segmenter/slf4j-api.jar:\
$STANFORD_POSTAGGER_PATH/stanford-postagger.jar:\
SF2017/stanford-ner/stanford-ner.jar:\
SF2017/stanford-parser/stanford-parser.jar:\
SF2017/stanford-parser/stanford-parser-3.7.0-models.jar"

segmenter = StanfordSegmenter(
    path_to_sihan_corpora_dict="SF2017/stanford-segmenter/data",
    path_to_model="SF2017/stanford-segmenter/data/pku.gz",
    path_to_dict="SF2017/stanford-segmenter/data/dict-chris6.ser.gz"
)

NERTagger = StanfordNERTagger('SF2017/stanford-ner/classifiers/chinese.misc.distsim.crf.ser.gz',
	encoding='utf-8')
ch_parser = StanfordParser(model_path='SF2017/stanford-parser/edu/stanford/nlp/models/lexparse/chineseFactored.ser.gz',
    encoding='utf-8')
#print([parse.tree() for parse in ch_parser.raw_parse(res)])
'''sentences = ch_parser.raw_parse_sents(res)
for line in sentences:
    for sentence in line:
        sentence.draw()'''


#res = segmenter.segment("美希迪波路治一般称作波路治，生于达尔贝达，摩洛哥职业足球运动员，现效力于美国职业足球大联盟球会科罗拉多急流。")
#print(type(res))
#print(res)
#for word, tag in NERTagger.tag(res.split( )):
#    print(word, tag)

#----------------------Load File Relation Corpus----------------------#
CORPUS = pickle.load(open("Src/CORPUSbyRelation1225.txt",'rb'))

for DOC in CORPUS['birthPlace']:
    corpus = list(CORPUS['birthPlace'][0:5])

