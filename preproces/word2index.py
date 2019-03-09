# -*- coding: utf-8 -*-
'''
Created on 2019年3月9日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#加载数据并生成索引和
import codecs 
import os 
import collections 
import pickle 
import numpy as np 
import re
import json
class TextLoader(): 
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'): 
        self.data_dir = data_dir 
        self.batch_size = batch_size 
        self.seq_length = seq_length 
        self.encoding = encoding 
        input_file = os.path.join(data_dir, "train_data.json") 
        vocab_file = os.path.join(data_dir, "vocab.pkl") 
        tensor_file = os.path.join(data_dir, "data.npy") 
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)): 
            print("reading text file") 
            self.preprocess(input_file, vocab_file, tensor_file) 
        else: 
            print("loading preprocessed files") 
            self.load_preprocessed(vocab_file, tensor_file) 
            self.create_batches() 
            self.reset_batch_pointer() # 当第一次训练时执行此函数，生成data.npy和vocab.pkl 
    def preprocess(self, input_file, vocab_file, tensor_file): 
        with open(input_file, "r",encoding=self.encoding) as f: 
            print()
            data = [json.loads(i) for i in f.readlines()]
#             print(data)
            all_words=[]#所有的词组
            words_array=[]#句子矩阵
            for line in data:
                all_words+=[j['word'] for j in line['postag']]
                words_array.append([j['word'] for j in line['postag']])
#             all_words
#             print(all_words)
                
            counter = collections.Counter(all_words) # 按键值进行排序 
            count_pairs = sorted(counter.items(), key=lambda x: -x[1]) # 得到所有的字符 
            count_pairs=count_pairs
            self.chars, _ = zip(*count_pairs) 
            self.vocab_size = len(self.chars) # 得到单词的索引，这点在文本处理的时候是值得借鉴的 
            print(self.vocab_size)
            self.vocab = dict(zip(self.chars, range(len(self.chars)))) # 将字符写入文件 
            with open(vocab_file, 'wb') as f: 
                pickle.dump(self.chars, f) # 使用map得到input文件中379591个字符对应的索引 
            
            self.tensor = np.array(list(map(self.vocab.get, data)))
#             print(self.tensor) 
#             np.save(tensor_file, self.tensor) 
    # 如果不是第一次执行训练，那么载入之前的字符和input信息 
    def load_preprocessed(self, vocab_file, tensor_file): 
        with open(vocab_file, 'rb') as f: 
            self.chars = pickle.load(f) 
            self.vocab_size = len(self.chars) 
            self.vocab = dict(zip(self.chars, range(len(self.chars)))) 
            self.tensor = np.load(tensor_file) 
            self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length)) 
    def create_batches(self): # tensor_size = 263336 batch_size = 50, seq_length = 100 # num_batches = 223 
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length)) # When the data (tensor) is too small, # let's give them a better error message 
        if self.num_batches == 0: 
            assert False, "Not enough data. Make seq_length and batch_size small." 
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length] 
        xdata = self.tensor 
        ydata = np.copy(self.tensor) # ydata为xdata的左循环移位，例如x为[1,2,3,4,5]，y就为[2,3,4,5,1] # 因为y是x的下一个字符
        ydata[:-1] = xdata[1:] 
        ydata[-1] = xdata[0] # x_batches 的 shape 就是 223 × 50 × 100 
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1) 
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1) 
    def next_batch(self): 
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer] 
        self.pointer += 1 
        return x, y 
    def reset_batch_pointer(self): 
        self.pointer = 0
if __name__ == '__main__':
    data_dir ='../data'
    tensor_size = 263336 #实际为1115394，1115000为取整之后的结果
    batch_size = 50
    seq_length = 100
    num_batches = 223
    textload=TextLoader(data_dir, batch_size, seq_length, encoding='utf-8')
    