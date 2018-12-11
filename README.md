# chinese-dialect-recognition
## Background:
For this challenge, a database covering China's 10 major dialects were provided include Changsha Dialect, Hebei Dialect, Nanchang Dialect, Shanghai Dialect, Fujian Dialect and Kejia Dialect,Ningxia Dialect,Hefei Dialect,Sichuan Dialect and Shan3xi Dialect. In this task, challengers were required to build a system that automatically identifies and assorts the audio files with different durations ( >3s for the task) provided in the challenge. 

## Network
![image](https://github.com/Colt1990/chinese-dialect-recognizaiton/blob/master/image/network.png)

## Feature
The following are the parameters used for HTK tools to obtain the FilterBank feature from the raw PCM files(16000Hz，16bit).
OURCEFORMAT = NOHEAD  
SOURCERATE = 625  
HEADERSIZE = 44  
TARGETKIND = FBANK  
TARGETRATE = 80000.0  
ZMEANSOURCE = T  
WINDOWSIZE = 200000.0  
USEHAMMING = T  
PREEMCOEF = 0.97  
NUMCHANS = 40  
LOFREQ = 0  
HIFREQ = 5500  
USEPOWER = F  
NUMCEPS = 12  
CEPLIFTER = 22  
SAVEWITHCRC = F  

## What I learned from this challenge
・PLP feature is quite useful for dialect recognition. Although only MFCC and FilterBank features were tried in this work and FilterBank showed better performance than MFCC, PLP may give the best performance from the results of other contestants.
・VAD processing and FilterBank feature with only <5.5KHz frequencies used are quite useful.  
・It is worth a try to use the powerful CNN network Resnet. Although I tried VGG16, failing to get a good performance.   

## Requirments
python 2.29   
pytorch 0.4.0  
webrtcvad  
wave 


## Reference
1. http://yerevann.github.io/2016/06/26/combining-cnn-and-rnn-for-spoken-language-identification/
