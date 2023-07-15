# Music Recommendation Based on Facial Emotion Recognition Using MTCNN and VGG-Face

## Introduction
This is a TensorFlow implementation for Music Recommendation based on Facial Emotion Recognition Using MTCNN and VGG-Face.

![image](https://github.com/byunghyun23/facial-emotion/blob/main/assets/fig1.png)
![image](https://github.com/byunghyun23/facial-emotion/blob/main/assets/fig2.png)

## Dataset
For training the model, you need to download this [link1](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=82).  
There are 7 emotions.  
We used 6 emotions: happy, sad, angry, anxious, hurt, and neutral.

The number of data for each emotion is as follows.
| emotion(kor)  | emotion(eng)  | size          |
| ------------- | ------------- | ------------- |
| 기쁨          | happy         | 15000         |
| 슬픔          | sad           | 15000         |
| 분노          | angry         | 15000         |
| 불안          | anxious       | 15000         |
| 상처          | hurt          | 15000         |
| 중립          | neutral       | 15000         |

## Preprocessing
We obtained the face in the image using MTCNN.
Also, we resized the size of the image to (224, 224) to use it as an input to the VGG-Face model.
The output of the VGG-Face model is facial features, which are used for learning.
You can do this by running
```
python preprocessing.py
```
Before running, the directory structure must be as follows.
```
--processed
--기쁨
--슬픔
--분노
--불안
--상처
--중립
기쁨.json
슬픔.json
분노.json
불안.json
상처.json
중립.json
```
The preprocessed data will be stored in the 'processed' directory.

## Train
```
python train.py
```
After training, the following Facial Emotion Recognition model and One-Hot enocoder is generated.
```
--my_model.h5
```
One-hot encoder is used to convert emotions.

## Predict
You can get the emotion of an image by running
```
python predict.py --file_name file_name
```

## Demo
Also, you can also use the model using Gradio by running
```
python web.py
```
![image](https://github.com/byunghyun23/facial-emotion/blob/main/assets/fig3.png)

