# Music Recommendation Based on Facial Emotion Recognition Using MTCNN and VGG-Face

## Introduction
This is a TensorFlow implementation for Music Recommendation based on Facial Emotion Recognition Using MTCNN and VGG-Face.

![image](https://github.com/byunghyun23/facial-emotion/blob/main/assets/fig1.png)
![image](https://github.com/byunghyun23/facial-emotion/blob/main/assets/fig2.png)

## Dataset
For training the model, you need to download [this](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=82).  
We used 6 emotions: happy, sad, angry, anxious, hurt, and neutral.

The number of data for each emotion is as follows.
| emotion(kor)  | emotion(eng)  | size          |
| ------------- | ------------- | ------------- |
| 기쁨          | happy         | 14,830         |
| 슬픔          | sad           | 14,871         |
| 분노          | angry         | 14,980         |
| 불안          | anxious       | 15,221         |
| 상처          | hurt          | 15,144         |
| 중립          | neutral       | 15,429         |

## Preprocessing
We obtained the cropped face images from dataset using MTCNN.  
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
my_model.h5
```
One-hot encoder is used to convert emotions.

## Predict
You can get the emotion of input image by running
```
python predict.py --file_name file_name
```

## Music Recommendation
We use Spotipy to get music recommendations based on emotion.  
When using Spotify, you must issue ID and Secret and fill it out below. (music.py)
```python
# Set up your Spotify API credentials
self.client_id = 'YOUR_ID'
self.client_secret = 'YOUR_SECRET'
```
To recommend a variety of music, half of the keywords for each emotion are randomly selected and used for recommendation.  

Keywords for emotions are as follows.
| emotion(kor)  | emotion(eng)  | keyword(eng)          |
| ------------- | ------------- | ------------- |
| 기쁨          | happy         | happy, delighted, glad, exiting, pleased, energetic, cheerful, satisfied, fulfilled, overjoyed |
| 슬픔          | sad           | sad, unhappy, tearful, gloomy, depressed, dejected, heartbroken, sorrowful, hurt, disappointed |
| 분노          | angry         | angry, mad, upset, furious, livid, irritated, enraged, incensed, outburst, resentful |
| 불안          | anxious       | anxious, nervous, uneasy, tense, worried, apprehensive, edgy, restless, uncertain, shaky |
| 상처          | hurt          | hurt, injured, damaged, wounded, scarred, sore, bruised, painful, aching, afflicted |
| 중립          | neutral       | neutral, emotionless, unfeeling, cold, unresponsive, impassive, apathetic, stoic, indifferent, unbiased |

## Demo
Also, you can also use the model using Gradio by running
```
python web.py
```
![image](https://github.com/byunghyun23/facial-emotion/blob/main/assets/fig3.png)
![image](https://github.com/byunghyun23/facial-emotion/blob/main/assets/fig4.png)

