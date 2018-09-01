# -*- coding: utf-8 -*-
'''
Created on 2019年3月9日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import multiprocessing
import json
import codecs
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
input_file = ['../data/train_data.json','../data/dev_data.json']
output_file='../data/train_sentence.txt'
# model = Word2Vec(size=32, window=5, min_count=5,workers=multiprocessing.cpu_count())
def sentence(input_file):
    for i in input_file:
        with open(i, "r",encoding='utf-8') as f: 
                    data = [json.loads(i) for i in f.readlines()]
        #             print(data)
                    all_words=[]#所有的词组
                    words_array=[]#句子矩阵
                    for line in data:
                        l=''
                        for j in line['postag']:
                            l+=j['word']+' '
                        l=l[0:-1]+'\n'
                        with codecs.open(output_file,"a",encoding='utf-8') as w: 
                            w.write(l)
def sentence_target(input_file,target_entity_file):
    with open(input_file, "r",encoding='utf-8') as f: 
            data = [json.loads(i) for i in f.readlines()]
#             print(data)
            all_words=[]#所有的词组
            words_array=[]#句子矩阵
            for line in data:
                l=''
                for j in line['spo_list']:
                    l+=j['object']+' '+j['subject']+' '
                l=l[0:-1]+'\n'
                with codecs.open(target_entity_file,"a",encoding='utf-8') as w: 
                    w.write(l)
 

# import logging
# # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# # model = Word2Vec(LineSentence(vocab), size=32, window=5, min_count=5,workers=multiprocessing.cpu_count())
# # model.save('../model/word2_model')
# model = Word2Vec.load('../model/word2_model')
# # print(model.wv.vocab)
# print(model.wv['男人'])
# print(model.most_similar('》'))
# print(model.wv.most_similar(positive=['军事', '政治'], negative=['经济']))
target_train_file='../data/train_data.json'
target_test_file='../data/dev_data.json'
target_train_entity='../data/train_entity.txt'
target_test_entity='../data/test_entity.txt'
target_train_class='../data/train_class.txt'
target_test_class='../data/test_class.txt'
sentence_target(target_train_file,target_train_entity)
sentence_target(target_test_file,target_test_entity)

