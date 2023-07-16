import gradio as gr
import face_detection, model, music
import pickle


def upload_image(image):
    try:
        image = detector.get_face(image)
        emotion = my_model.predict(image)

        emotion = encoder.inverse_transform(emotion)
        recommend = music_model.get_music(kor_emotion=emotion[0][0], gradio=True)

        return recommend
    except Exception as e:
        return "Sorry, I can't recognize your face.\n"


### Setting
detector = face_detection.Mtcnn()
model_name = 'my_model.h5'
my_model = model.MyModel()
my_model.set_model(model_name)

encoder_name = 'encoder.pkl'
with open(encoder_name, 'rb') as f:
    encoder = pickle.load(f)

music_model = music.Music()


### Gradio
title = 'Music Recommendations based on Facial Recognition'
description = 'Ref: https://github.com/byunghyun23/facial-emotion'
image_input = gr.components.Image(label='Input image', type='numpy')
output_text = gr.components.Textbox(label='Recommendation')
custom_css = '#component-12 {display: none;} #component-1 {display: flex; justify-content: center; align-items: center;} img.svelte-ms5bsk {width: unset;}'

iface = gr.Interface(fn=upload_image, inputs=image_input, outputs=output_text,
                     title=title, description=description, css=custom_css)
iface.launch(server_port=8080)