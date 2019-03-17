# -*- coding: utf-8 -*-
'''
Created on 2019年3月17日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word
def load_pos_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/pos_vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word
def create_data(source_sents, target_sents,pos_sents): 
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    pos2idx,idx2pos=load_pos_vocab()
    
    # Index
    x_list,x_pos_list, y_list, Sources, Source_pos,Targets = [], [], [], [],[],[]
    for source_sent, target_sent,pos_sent in zip(source_sents, target_sents,pos_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()] 
        x_pos=[pos2idx.get(word, 1) for word in (pos_sent + u" </S>").split()] 
        if max(len(x_pos),max(len(x), len(y))) <=hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            x_pos_list.append(np.array(x_pos))
            Sources.append(source_sent)
            Targets.append(target_sent)
            Source_pos.append(pos_sent)
    
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    X_pos=np.zeros([len(x_pos_list), hp.maxlen], np.int32)
    for i, (x, y,x_pos) in enumerate(zip(x_list, y_list,x_pos_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
        X_pos[i]=np.lib.pad(x_pos, [0, hp.maxlen-len(x_pos)], 'constant', constant_values=(0, 0))
    return X,X_pos, Y, Sources, Targets,Source_pos

def load_train_data():
    de_sents = [line for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [line for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    pos_sents= [line for line in codecs.open(hp.source_train_pos, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    X,X_pos, Y, Sources, Targets,source_pos = create_data(de_sents, en_sents,pos_sents)
    return X, X_pos,Y
    
def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
#         line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    pos_sents= [line for line in codecs.open(hp.source_test_pos, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"] 
    X, X_pos,Y, Sources, Targets,Source_pos = create_data(de_sents, en_sents,pos_sents)
    return X, X_pos,Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X, X_pos,Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    X_pos=tf.convert_to_tensor(X_pos, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X,Y])#X_pos
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,#,x_pos
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T),(N,T) (N, T), (),x_pos
