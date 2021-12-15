# MeloGraph
This repository holds code for the  MUSI-6201 (Audio Content Analysis) project submission of group 3 (Noel Alben, Rhythm Jain and Thiago Roque).
\
\
**Goal:** Automatic Note Segmetation of Monophonic Melodies and Graph Representation of the Note-to-Note Sequences

## Working Example 
Input Melody
![alt text](https://github.com/nol-alb/melograph_submission/blob/main/images/Transcript.png)
Graph Output
![alt text](https://github.com/nol-alb/melograph_submission/blob/main/images/Graph.png)



## Code Organization 
- Baseline_Implementation contains code from our Baseline Report
- The Onsetdet.py and Pitch_Tracking.py contains source code for our Onset Detection and Pitch Tracking implementations

## Running the Web Applicaton
The Web Application is Self Contained within the directory WebApp
```
cd ..<Clone Repository>
pip install -r requirements.txt
cd WebApp
export FLASK_APP=app
flask run
```






