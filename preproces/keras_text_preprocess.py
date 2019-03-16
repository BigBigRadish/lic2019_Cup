# -*- coding: utf-8 -*-
'''
Created on 2019年3月14日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#使用keras进行文本预处理
import logging
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import codecs
import numpy as np
from keras.preprocessing import text, sequence
with codecs.open('../data/train_sentence.txt','r',encoding='utf-8') as f:
    sentences=f.readlines()
# sentence=LineSentence('../data/train_sentence.txt')

train_sentence=sentences[:173109]
test_sentence=sentences[173109:]
# 生成的字典取频数高的max_feature个word对文本进行处理，其他的word都会忽略
# 每个文本处理后word长度为maxlen
maxlen = 50
# 词嵌入长度取facebook fasttext的300长度
embedding_size = 300

# 使用keras进行分词，word转成对应index
tokenizer = text.Tokenizer(filters='\t\n')#默认全部转换成小写
tokenizer.fit_on_texts(sentences)
train_feature = tokenizer.texts_to_sequences(train_sentence)
test_feature = tokenizer.texts_to_sequences(test_sentence)
# 将每个文本转成固定长度maxlen，长的截取，短的填充0
train_feature = sequence.pad_sequences(train_feature, maxlen)
test_feature = sequence.pad_sequences(test_feature, maxlen)
print(train_feature[0:5])

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# model = Word2Vec(LineSentence('../data/train_sentence.txt'), size=300, window=5, min_count=1,workers=multiprocessing.cpu_count())
# model.save('../model/word2_model')
model = Word2Vec.load('../model/word2_model')
# print(model.wv.vocab)
print(model.most_similar('》'))
print(model.wv['籍贯'])
# 词嵌入向量转dict的
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embedding_index = model.wv
  
# 文本中词的index映射对应的词嵌入
word_index = tokenizer.word_index
nb_words =  len(word_index) # 基于文本的词典总长为len(word_index)，
print(nb_words)
# 由于使用max_feature进行了筛选，则最终使用的词典由max_feature, len(word_index)决定
embedding_matrix = np.zeros((nb_words+1, embedding_size))
for word, i in word_index.items():
#     print(word,i)
    try:
#     if i>= max_feature: continue #如果i大于max_feature,说明该词已经超出max_feature范围，无需处理
    
        embedding_vector = embedding_index[word]
    except KeyError:
        print(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector
