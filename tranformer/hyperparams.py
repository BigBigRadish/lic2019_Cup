# -*- coding: utf-8 -*-
'''
Created on 2019年3月17日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = '../data/train_sentence_1.txt'
    target_train = '../data/train_entity.txt'
    source_test = '../data/test_sentence.txt'
    target_test = '../data/test_entity.txt'
    source_train_pos='../data/train_pos.txt'
    source_test_pos='../data/test_pos.txt'
    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    
    # model
    maxlen = 80 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.