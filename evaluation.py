
import soundfile as sf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import os
import gc
from scipy.signal import get_window
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers, models
import random
import tensorflow.keras as keras
import xlsxwriter
import datetime
'''initiation of variable and output'''
wb = xlsxwriter.Workbook('output11_model_model_musan_lib_vox_aishell_ted.xlsx')
model_name='model_musan_lib_vox_aishell'
class_lable = ['music','speech']
num_lable=2
num_mfcc=40
'''directory of test file
test/folder/file
'''
duration=12
time_index=5
directory='test'
time=[]
model =  keras.models.load_model(model_name)
'''evaluation of file in test'''
for folder in os.listdir(directory):
    sheet = wb.add_worksheet(folder)
    sheet.write(1,3,"=COUNTIF(B:B,"+"\"speech\""+")")
    sheet.write(2,3,"=COUNTIF(B:B,"+"\"music\""+")")
    a=0
    for file in os.listdir(directory+'/'+folder):
        z, sr = sf.read(directory+'/'+folder+'/'+file)
        y, index = librosa.effects.trim(z,top_db=60)
        intervals = librosa.effects.split(y,top_db=60,frame_length=512,hop_length=256)
        v=y[intervals[0][0]:intervals[0][1]]
        for seg in range(np.size(intervals,0)-1):
            temp = y[intervals[seg+1][0]:intervals[seg+1][1]]
            v=np.append(v,temp)
        c=np.array(np.zeros(num_mfcc*duration))
        sample=[]
        predict=[]
        b=[0,0]
        #if the file less than 3 seconds, asuume as silence
        if(len(y)/sr<1 ):
            print(file)
            sheet.write(a, 0, file)
            print("0,0")
            print("silence")
            sheet.write(a, 1, "silence") 
            sheet.write(a,2,"0,0")
            time.append(2)
            a=a+1
            continue
        mfccs = librosa.feature.mfcc(y=y, n_mfcc=num_mfcc, sr=sr,n_fft=512,hop_length=256, window = get_window("hamming",512),n_mels = 64,fmax=sr//2)
        #print(np.shape(mfccs[:,0]))
        '''
        if number of frame not multiply of 12
        copy the last 12 frame
        erase the frame from last to make it 12*n
        '''
        if len(mfccs[0])%duration!=0:
            
            #temp=mfccs[:,-duration:]
            #print(temp.shape)
            mfccs=mfccs[:,0:(len(mfccs[0])-len(mfccs[0])%duration)]
            #print(mfccs.shape)
            #mfccs=np.append(mfccs,temp,axis=1)
            #print(len(mfccs[0]))
        long=len(mfccs[0])//duration
        new_mfccs=mfccs.reshape(num_mfcc*duration,int(long))
        #print(new_mfccs.shape)
        '''make prediction of each array'''
        for i in range(len(new_mfccs[0])):
            c=new_mfccs[:,i]
            #if((c_prev.all==c.all)):
                #print("nou")
                #exit()
            #print(c.shape)
            prediction = model.predict([tf.expand_dims(c,axis=0)])
            b[np.argmax(prediction)]= b[np.argmax(prediction)]+1
            predict.append(np.argmax(prediction))
        if not os.path.exists('plot/'+folder):
                os.makedirs('plot/'+folder)
        if not os.path.exists('plot/'+folder+'/'+file[:-4]+'.png'):
            fig= plt.figure()
            plt.subplot(2,1,1)
            plt.plot(v)
            plt.subplot(2,1,2)
            plt.step(np.arange(0,np.size(predict),1),predict)
            plt.savefig('plot/'+folder+'/'+file[:-4]+'.png')
            plt.figure().clear()
            plt.close('all')
            plt.clf()
        '''output the raw result'''
        print(file)
        sheet.write(a, 0, file)
        print(b)
        print(class_lable[np.argmax(b)])
        sheet.write(a, 1, class_lable[np.argmax(b)]) 
        sheet.write(a,2,str(b[0])+","+str(b[1]))
        time.append(np.argmax(b))
        a=a+1

'''
process the raw result
if it is less than 3 consetative frame equal
ignor
output the time code which music change to speech or vice versa
'''
f = open("result.txt", "w")
print(len(time))
for i in range(1,len(time)-2):
    if(time[i]==2):
        continue
    if((time[i]!=(time[i+1]))and (time[i]!=(time[i-1]))):
        time[i]=time[i-1]
for i in range(0,len(time)-1):
    if(time[i]==2):
        continue
    if(i==0):
        print(str(i)+','+str(datetime.timedelta(seconds=(i*time_index)))+','+class_lable[time[i]])
        f.writelines(str(i)+','+str(datetime.timedelta(seconds=(i*time_index)))+','+class_lable[time[i]]+'\n')
        continue
    if(time[i]!=(time[i-1])):
        print(str(i)+','+str(datetime.timedelta(seconds=(i*time_index)))+','+class_lable[time[i]])
        f.writelines(str(i)+','+str(datetime.timedelta(seconds=(i*time_index)))+','+class_lable[time[i]]+'\n')
        
wb.close()