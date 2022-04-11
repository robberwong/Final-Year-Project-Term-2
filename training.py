import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from scipy.signal import get_window
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers, models
import tensorflow as tf
import random
import tensorboard
import datetime
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

log_folder = 'logs'
class_lable = {'music':0, 'speech':1}
duration=12 #modify this number to have more frame in one group
data= 'data/'
sample=[]
num_lable=2
num_mfcc=40 #modify this number to have more MFCC in one frame
'''model'''
model_name='model_musan_origin'
if not os.path.exists(model_name+'/'):
                
    model = tf.keras.Sequential([
        layers.Input(shape=(num_mfcc*duration,)),
        layers.Reshape((num_mfcc,duration,1)),
        #preprocessing.Resizing(13, 12),
        layers.Conv2D(32, 3, padding='valid', activation='relu'),
        layers.MaxPooling2D(),
        preprocessing.Normalization(),
        layers.Conv2D(64, 3, padding='valid',activation='relu'),
        layers.MaxPooling2D(),
        preprocessing.Normalization(),
        #layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2,activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )
else:
    model =  tf.keras.models.load_model(model_name)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

x=np.load('x_musan_origin.npy') #load the processed data from preprocessing.py
y=np.load('y_musan_origin.npy') #load the processed data from preprocessing.py
x = tf.stack(x)
y = tf.stack(y)

y_train=y.numpy()
#print(x[0])
#print(x[1])
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=0, patience=2)
print(y.shape)
print(np.shape(y_train))
class_weights = class_weight.compute_class_weight(class_weight =  'balanced', classes= np.unique(y_train), y = y_train)
#class_weights = dict(zip( np.unique(y_train), class_weights)),
class_weight = {0: class_weights[0], 1: class_weights[1]}
print(class_weight)
history =model.fit(x,y, epochs=10,callbacks=[es],validation_split=0.05,class_weight=class_weight,shuffle=True )
model.summary()

model.save(model_name+'/')



