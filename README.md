# chinese-dialect-recognition
## Background:
For this challenge, a database covering China's 10 major dialects will be open. Challengers shall take both or either of two tasks with different difficulties.  In each task, challengers are required to build a system that automatically identifies and assorts the audio files with different durations (≤3s for the first task and >3s for the second task) provided in the challenge. The final ranking will be decided based on the classification accuracy of your system.  Training set and development set is open for challengers' use but the test set is not open to the public.

## Network
This code was revised based on the baseline code offered by iFLYTEK. 
![image](https://github.com/Colt1990/chinese-dialect-recognizaiton/blob/master/image/network.png)

## What I learned from this challenge
・PLP feature is quite useful for dialect recognition. Although only MFCC and FilterBank features were tried in this work and FilterBank showed better performance than MFCC, PLP may give the best performance from the results of other contestants.
・VAD processing and FilterBank feature with only <5.5KHz frequencies used are quite useful.

・It is worth a try to use the powerful CNN network Resnet. Although I tried VGG16, the result was not good.   

## Requirments
python 2.29 

pytorch 0.4.0 

webrtcvad

wave 
