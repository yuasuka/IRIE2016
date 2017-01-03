# -*- coding: utf-8 -*-

import csv
import pickle

import jieba
from tqdm import *

#----------------Read File----------------#
corpus = pickle.load(open("Corpus1225.txt",'rb'))
