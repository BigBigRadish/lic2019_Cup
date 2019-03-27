# -*- coding: utf-8 -*-
'''
Created on 2019年3月24日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import bilsm_crf_model
import keras
import time
from tensorflow.keras.callbacks import TensorBoard
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logdir', 
                                         histogram_freq= 0, 
                                         write_graph=True, 
                                         write_images=True)
EPOCHS = 10
model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
# train model
model.fit(train_x, train_y,batch_size=64,epochs=EPOCHS, callbacks=[tbCallBack],validation_data=[test_x, test_y])
model.save('model/crf.h5')
