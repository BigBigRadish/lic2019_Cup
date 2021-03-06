# -*- coding: utf-8 -*-
'''
Created on 2019年3月24日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import numpy
import codecs
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform
import json
raw_data_file=['./data/train_data.json','./data/dev_data.json']
def tag_data(raw_data_file,tag_train_data): #bio标记数据
    with open(raw_data_file,'r',encoding='utf-8') as f:
        data = [json.loads(i) for i in f.readlines()]
        for line in data:#遍历每一行
            text=line['text']
            text_list=list(text)
#             print(text_list)
            word_dic=[]#初始化BIO tag
            for j in text_list:
                if j!='《' and j!='》':
                    word_dic.append('O')
                else:
                    word_dic.append('o-tag')
            spo_bj={}#obj字典
            for i in line['spo_list']:#将实体类型以及实体对都先加入临时字典
                spo_bj[i['object_type']]=i['object']
                spo_bj[i['subject_type']]=i['subject']
            for k,v in spo_bj.items():
#                 print(k,v)
#                 print(text)
                start_pst=text.find(v)
#                 print(text[start_pst:len(v)+start_pst])               
#                 print(start_pst)
                word_dic[start_pst]='B-'+str(k)
                for m in range(start_pst+1,len(v)+start_pst):
                    word_dic[m]='I-'+str(k)
#             print(word_dic)
            with codecs.open(tag_train_data,'a',encoding='utf-8') as f:
                for i in range(0,len(text_list)):
                    f.write(text[i]+' '+word_dic[i]+'\r\n')
                f.write('\r\n\r\n')     
def load_data():
    schema_path='./data/all_50_schemas'
    train = _parse_data(open('./data/train_data_1.data', 'rb'))
    test = _parse_data(open('./data/test_data_1.data', 'rb'))
    word_counts = Counter(row[0] for sample in train for row in sample if sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    with codecs.open(schema_path,'r',encoding='utf-8') as f:
        data = [json.loads(i) for i in f.readlines()]
        tags=[]
        for line in data:
            tags.append('B-'+line["object_type"])
            tags.append('I-'+line["object_type"])
            tags.append('B-'+line['subject_type'])
            tags.append('I-'+line['subject_type'])
        tags.append('O')
        tags.append('o-tag')
        chunk_tags=list(set(tags))
        print(chunk_tags)
    # save initial config data
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags)


def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'

    string = fh.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    fh.close()
    return data

def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length

tag_data('./data/train_data.json','./data/train_data_1.data')