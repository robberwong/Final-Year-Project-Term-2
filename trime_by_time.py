import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from scipy.signal import get_window
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
file='temp/ted/Steven Sharp Nelson_ How to find peace with loss through music _ TED.mp4' #file directory of the video file
audio_file='audio/audio.wav' #directory to save the audio file extracted from video file
audio = AudioSegment.from_file(file)
audio=audio.set_channels(1)
audio = audio.set_frame_rate(16000)
audio.export(audio_file,format="wav")
results=open('result.txt')

index=[]
class_label=[]
trimmed='trimmed'

'''Audio process'''
z, sr = sf.read(audio_file)
for line in results.readlines(): 
    line=line.strip()                       
    line = line.split(',')                          
    index.append(int(line[0])*5*sr)
    class_label.append(line[2])
index.append(np.size(z))
print(index)
v=z[index[0]:index[1]]
for seg in range(np.size(index)-1):
    if(class_label[seg]=='speech'):
        temp = z[index[seg]:index[seg+1]]
        sf.write('for_transcript/'+str(index[seg])+'.wav', temp, sr)  #save each speech segment for each speech file.
    if(class_label[seg]=='music'):
        temp = z[index[seg]:index[seg+1]]
        sf.write('music_extracted/'+str(index[seg])+'.wav', temp, sr)  #save each speech segment for each speech file.
    

        