# Music Recommendation Based on Facial Emotion Recognition Using MTCNN and VGG-Face

## Introduction
This is a TensorFlow implementation for Music Recommendation based on Facial Emotion Recognition Using MTCNN and VGG-Face.

![image](https://github.com/byunghyun23/image-captioning/blob/main/assets/fig_1.png)
![image](https://github.com/byunghyun23/image-captioning/blob/main/assets/fig_2.png)

## Dataset
For training the model, you need to download this [link1](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=82).  
There are 7 emotions.  
We used 6 emotions: happy, angry, anxious, hurt, sad, and neutral.

## Preprocessing
We obtained the face in the image using MTCNN.
Also, we resized the image to (224, 224) to input  it as input for learning the captioning model.  
So you can label image features and captions and you can also get a dataset classified as training and test by running
```
python preprocessing.py
```
The generated files are:
```
--idx_to_word.pkl
--word_to_idx.pkl
--train_captions.pkl
--test_captions.pkl
--train_encoding.pkl
--test_encoding.pkl
```

## Train
```
python train.py
```
After training, the following model is created.
```
--caption_model.h5
```

## Predict
You can get the caption of an image by running
```
python predict.py --file_name file_name
```

## Demo
Also, you can also use the model using Gradio by running
```
python web.py
```
![image](https://github.com/byunghyun23/image-captioning/blob/main/assets/fig_3.PNG)

