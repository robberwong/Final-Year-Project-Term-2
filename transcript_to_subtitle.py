import io
import datetime

sub=open('subtitle.srt','w',encoding='utf8')
a=1
f=open('recognition.txt',encoding='utf8')
for line in f.readlines(): 
    line=line.strip()                       
    line = line.split(':')                          
    index=line[0]
    word=line[1]
    time=int(index)//16000
    start=datetime.datetime(1,1,1,0,0,0)+datetime.timedelta(seconds=time)
    end=datetime.datetime(1,1,1,0,0,0)+datetime.timedelta(seconds=time+10)
    print(str(start.time())+'-->'+str(end.time()))
    sub.writelines(str(a)+'\n'+str(start.time())+',000'+' --> '+str(end.time())+',000'+'\n'+word+'\n'+'\n')
    a=a+1
    