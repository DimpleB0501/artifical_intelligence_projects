# Task 1
### Task  1.1 The task  is to get all the landmarks on the body from the image . 
|Human pose estimation output|
|:------------:|
|![Output Image](./images/output_image.png)|

### Task 1.2 Elaborate on Technology  used. 
I used **Keypoint RCNN deep learning model with a ResNet-50 base architecture** available in pytorch. <br/>
This model has been pre-trained on the COCO Keypoint dataset.<br/>
It outputs the keypoints for 17 human parts and body joints. <br/>
They are: ‘nose’,  ‘left_eye’, ‘right_eye’, ‘left_ear’, ‘right_ear’, ‘left_shoulder’, ‘right_shoulder’, ‘left_elbow’, ‘right_elbow’, ‘left_wrist’, ‘right_wrist’, ‘left_hip’, ‘right_hip’, ‘left_knee’, ‘right_knee’, ‘left_ankle’, ‘right_ankle’.

# Task 2 
For this task, the input is live feed from a webcam.In this section, you should get only  the specified landmarks on the body.
The landmarks are:
- Nose
- Left shoulder 
- Right shoulder
- Right elbow
- Left elbow
- Right knee
- Left knee 
- Right ankle
- Left ankle

|Output pose estimation video|
|:------------:|
|![Output video](./images/task2.gif)|
|[Youtube Link](https://www.youtube.com/watch?v=IKSU_8M1veY)|

**Note**: Task 2 is implemented as a python file. To run the file<br/>
`cd ~/pose_estimation_assignment/pose_estimation/src`<br/>
 `python3 task2_keypoint_rcnn_videos.py `
 
# Task 3
Illustrate the process of creating a pose classification model with steps starting from ML package choice (tensorflow/pytorch) to testing the model.
What method can be used as a feedback control loop in pose classification if a human pose estimation model is used to get the landmarks ? In simple words, can landmarks of body parts be used to validate the result from the pose classifier?

