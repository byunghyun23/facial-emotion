import click
import numpy as np
import cv2
import face_detection, model, music
import pickle
import matplotlib.pyplot as plt


@click.command()
@click.option('--file_name', default='assets/sample.jpg', help='Input file name')
@click.option('--model_name', default='my_model.h5', help='Model name')
@click.option('--encoder_name', default='encoder.pkl', help='Encoder name')
def run(file_name, model_name, encoder_name):
    detector = face_detection.Mtcnn()

    image = np.fromfile(file_name, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    input_image = image.copy()
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    try:
        image = detector.get_face(image)
    except Exception as e:
        print("Sorry, I can't recognize your face.")
        return

    # detector = None

    my_model = model.MyModel()
    my_model.set_model(model_name)

    emotion = my_model.predict(image)

    with open(encoder_name, 'rb') as f:
        encoder = pickle.load(f)

    emotion = encoder.inverse_transform(emotion)
    print(emotion)

    music_model = music.Music()
    recommend = music_model.get_music(kor_emotion=emotion[0][0])
    print(recommend)

    plt.rc('font', family='Malgun Gothic')
    plt.imshow(input_image)
    plt.title(recommend)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    run()