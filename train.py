import click
import os
import cv2
import numpy as np
import model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random
import pickle
random.seed(42)


def load_data(model, images_dir, start=0, end=None):
    name_list = []
    image_list = []

    files = os.listdir(images_dir)
    random.shuffle(files)

    cnt = 0
    if end is None:
        end = len(files)
    for file in files[start:end]:
        try:
            path = os.path.join(images_dir, file)

            # image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = np.fromfile(path, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = model.processing(image)
            image = image[0]

            # name_list.append(file)
            name_list.append(file.split('_')[3])
            image_list.append(image)

            cnt += 1
            print(cnt, file, file.split('_')[3])
            del image, file

        except FileNotFoundError as e:
            print('ERROR : ', e)

    names = np.array(name_list)
    images = np.stack(image_list)

    return names, images


@click.command()
@click.option('--data_dir', default='data/processed', help='Data directory')
@click.option('--batch_size', default=8, help='Batch size')
@click.option('--epochs', default=300, help='Epochs')
@click.option('--encoder_name', default='encoder.pkl', help='Encoder name')
@click.option('--train_split', default=0.8, help='Train split')
def run(data_dir, batch_size, epochs, encoder_name, train_split):
    save_name = 'my_model'
    my_model = model.MyModel()

    checkpoint = ModelCheckpoint(save_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    callbacks_list = [checkpoint, es]

    y_train, X_train = load_data(my_model, data_dir)
    y_train = y_train.reshape(-1, 1)

    one_hot_encoder = OneHotEncoder()
    y_train = one_hot_encoder.fit_transform(y_train)
    y_train = y_train.toarray()
    print(y_train)
    print(X_train.shape, y_train.shape)
    with open(encoder_name, 'wb') as f:
        pickle.dump(one_hot_encoder, f)

    train_num = int(len(X_train) * train_split)
    print('train_num:', train_num)

    steps = int(train_num / batch_size)
    print('steps:', steps)

    history = my_model.train(X_train, y_train, train_num, batch_size, steps, epochs, callbacks_list)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_len = np.arange(len(train_loss))
    val_len = np.arange(len(val_loss))
    plt.plot(train_len, train_loss, marker='.', c='blue', label='Train-set Loss')
    plt.plot(val_len, val_loss, marker='.', c='red', label='Validation-set Loss')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    run()