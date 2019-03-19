# -*- coding: utf-8 -*-
'''
Created on 2019年3月19日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
import os
import json
import codecs
class LtpParser:
    def __init__(self):
        LTP_DIR = 'D:\LTP\MODEL\ltp_data'  # ltp模型目录的路径
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(LTP_DIR, "cws.model"))# 分词模型路径，模型名称为`cws.model`

        self.postagger = Postagger()
        self.postagger.load(os.path.join(LTP_DIR, "pos.model"))# 词性标注模型路径，模型名称为`pos.model`

        self.parser = Parser()
        self.parser.load(os.path.join(LTP_DIR, "parser.model"))# 依存句法分析模型路径，模型名称为`parser.model

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(LTP_DIR, "ner.model"))# 命名实体识别模型路径，模型名称为`ner.model`

        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(LTP_DIR, 'pisrl_win.model'))# 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。
    def ner(self,words, postags):
        netags = self.recognizer.recognize(words, postags)  # 命名实体识别
        #for word, ntag in zip(words, netags):
        #   print(word + '/' + ntag)
        self.parser.release()  # 释放模型
        return netags
# 命名实体识别
parser=LtpParser()
with open('./data/train_data.json','r',encoding='utf-8') as f:
    data = [json.loads(i) for i in f.readlines()]
    all_words=[]#所有的词组
    words_array=[]#句子矩阵
    for line in data:
        l=[]
        pos=[]
        for j in line['postag']:
            l.append(j['word'])
            pos.append(j['pos'])
        netags=parser.ner(l,pos)
        with codecs.open('ltp_ner.txt',"a",encoding='utf-8') as w: 
            tags = []
            dict = []
            
            for word, ntag in zip(l, netags):
                if(ntag != 'O'):#过滤非命名实体
                    tags.append(ntag)
                    if (ntag not in dict):
                        dict.append(ntag)
                    print(word + '/' + ntag)
                    w.write(word+' ')
            w.write('\t\n')
