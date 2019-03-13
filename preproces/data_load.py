# -*- coding: utf-8 -*-
'''
Created on 2019年3月11日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#加载数据集
'''
 fpath1, fpath2 分别是 原始输入文件 and 目标文件

'''
import collections 
import pickle
import tensorflow as tf
from utils import calc_num_batches
import codecs
import numpy as np
def load_vocab(vocab_fpath):

    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>
    Returns
    two dictionaries.
    '''
    with codecs.open(vocab_fpath,'rb') as f:
        vocab=pickle.load(f)
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
#     print(token2idx)
    return token2idx, idx2token
def generate_vocab(input_file, vocab_file): 
        with codecs.open(input_file, "rb",encoding='utf-8') as f: 
            data = f.read().split() # 统计出每个字符出现多少次，统计结果总共有65个字符，所以vocab_size = 65
            print(len(data)) 
            counter = collections.Counter(data) # 按键值进行排序 
            count_pairs = sorted(counter.items(), key=lambda x: -x[1]) # 得到所有的字符 
            count_pairs=count_pairs
            chars, _ = zip(*count_pairs) 
            vocab = dict(zip(chars, range(1,len(chars)))) # 将字符写入文件 
#             print(vocab)
#             vocab_size = len(chars) # 得到字符的索引，这点在文本处理的时候是值得借鉴的 
#             print(vocab_size)
            with codecs.open(vocab_file, 'wb') as f1: 
                pickle.dump(vocab, f1) # 使用map得到input文件中1115394个字符对应的索引 
def load_data(fpath1, fpath2, maxlen1, maxlen2):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    with open(fpath1, 'r',encoding='utf-8') as f1, open(fpath2, 'r',encoding='utf-8') as f2:
        for sent1, sent2 in zip(f1.readlines(), f2.readlines()):
#             print(sent1.strip())
            if len(sent1.split()) + 1 > maxlen1: continue # 1: </s>
            if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
            sents1.append(sent1)
            sents2.append(sent2)
#     print(sents1)
    return sents1, sents2


def encode(inp, type1, dict1):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1维byte数组.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary
    Returns
    list of numbers
    '''
#     inp_str = inp.decode("utf-8")
    tokens = inp.split()
    print(tokens)
    x = [dict1.get(t) for t in tokens]
    print(x)
    return x

def generator_fn(sents1, sents2, vocab_fpath):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
#     print(sents1)
    token2idx, _ = load_vocab(vocab_fpath)
    print(token2idx)
    for sent1, sent2 in zip(sents1, sents2):
        print(sent1)
        x = encode(sent1, "x", token2idx)
        y = encode(sent2, "y", token2idx)
#         decoder_input, y = y, y[1:]
        x_seqlen, y_seqlen = len(x), len(y)
        yield (np.array(x), x_seqlen), ( np.array(y), y_seqlen)#decoder_input,, sent1)


def input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=False):
    '''生成一批一批的数据
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean
    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    shapes = (([None],()),
              ([None],()))
    types = ((tf.int32, tf.int32), #, tf.string),
             (tf.int32, tf.int32))#, tf.string))
    paddings = ((-1, -1,),
                (-1, -1,))
    print(paddings)
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_types=types,
#         (tf.int32, tf.int32),
        output_shapes=shapes)
#         (tf.TensorShape([None]), tf.TensorShape([])))  # <- arguments for generator_fn. converted to np string arrays
#     print(dataset)
#     if shuffle: # for training
#         dataset = dataset.shuffle(128*batch_size)
#     print(dataset)
    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)
    
    return dataset



def get_batch(fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean
    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)
    batches = input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=shuffle)
    print(batches)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)
if __name__ == '__main__':
    fpath='../data/train_sentence.txt'
    train_pos_path='../data/train_pos.txt'#train pos
    vocab_file='../data/vocab.pkl'
    pos_vacab_file='../data/pos_vocab.pkl'
    train_entity_file='../data/train_entity.txt'
#     generate_vocab(train_pos_path, pos_vacab_file)
#     with codecs.open(pos_vacab_file, 'rb') as f1: 
#                 vocab=pickle.load(f1) # 使用map得到input文件中1115394个字符对应的索引
#     print(vocab)
    get_batch(fpath, train_entity_file, 50, 50, vocab_file, 50, shuffle=False)
#     load_vocab(vocab_file)