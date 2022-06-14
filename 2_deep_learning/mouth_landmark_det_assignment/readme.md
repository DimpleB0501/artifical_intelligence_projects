# Real-time mouth recognition 
In the video below, real-time mouth recognition is implemented. Two red dots appear on the edge of the live mouth. And an effect is applied when the mouth is opened wider than a threshold.

|Real time mouth recognition|
|:------------:|
|![Output video](./images/landmark.gif)|
|[Youtube Link](https://youtu.be/J-6L2AwqSf0)|

### Running the code
Run in terminal <br/>
`python3 main_webcam.py`

# Notes
I have used [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) implementation to detect 68 landmarks on the face. <br/>
From those landmarks, I have used mouth landmarks for the implementation. 
![khan](./images/lip_landmark.jpg)
