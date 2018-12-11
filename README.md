# chinese-dialect-recognition
## Background:
For this challenge, a database covering China's 10 major dialects will be open. Challengers shall take both or either of two tasks with different difficulties.  In each task, challengers are required to build a system that automatically identifies and assorts the audio files with different durations (≤3s for the first task and >3s for the second task) provided in the challenge. The final ranking will be decided based on the classification accuracy of your system.  Training set and development set is open for challengers' use but the test set is not open to the public.

## Network
This code was revised based on the baseline code offered by iFLYTEK. 
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

・It is worth a try to use the powerful CNN network Resnet. Although I tried VGG16, the result was not good.   

## Requirments
python 2.29 

pytorch 0.4.0 

webrtcvad

wave 


## Reference
1. http://yerevann.github.io/2016/06/26/combining-cnn-and-rnn-for-spoken-language-identification/
