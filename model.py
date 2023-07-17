# pip install keras_vggface
# pip install keras-applications
from keras.engine import Model
from keras.layers import Flatten, Dense
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
from tensorflow.python.keras.models import load_model
from tensorflow.keras import regularizers
import tensorflow as tf


class MyModel:
    def __init__(self):
        # Customize parameters
        self.nb_class = 6
        self.hidden_dim = 512

        # Load the VGG-Face model
        # vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3))
        vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
        # vgg_model.summary()

        # Fine-tuning
        # last_layer = vgg_model.get_layer('pool5').output
        # x = Flatten(name='flatten')(last_layer)
        # x = Dense(self.hidden_dim, activation='relu')(x)
        # x = Dense(self.hidden_dim, activation='relu'(x)
        # out = Dense(self.nb_class, activation='softmax')(x)
        last_layer = vgg_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(self.hidden_dim*2, activation='relu')(x)
        x = Dense(self.hidden_dim, activation='relu', kernel_regularizer=regularizers.l1(0.001))(x)
        x = Dense(self.hidden_dim, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        out = Dense(self.nb_class, activation='softmax')(x)

        adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.custom_vgg_model = Model(vgg_model.input, out)
        self.custom_vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

    def __batch_generator(self, X_train, Y_train, batch_size):
        while True:
            steps = int(len(X_train) / batch_size)

            for step_index in range(steps):
                start = step_index * batch_size
                end = start + batch_size
                x_data_batch = X_train[start:end]
                y_data_batch = Y_train[start:end]
                x_train_batch = np.asarray(x_data_batch)
                y_train_batch = np.asarray(y_data_batch)

                # print(y_train_batch, y_train_batch.shape)

                yield x_train_batch, y_train_batch

    def processing(self, image):
        image = image.astype('float32')
        image = np.expand_dims(image, axis=0)
        # image = preprocess_input(image, version=1)
        image = preprocess_input(image, version=2)

        return image

    def set_model(self, model_name):
        self.custom_vgg_model = load_model(model_name)

    def train(self, X_train, y_train, train_num, batch_size, steps, epochs, callbacks_list):
        train_generator = self.__batch_generator(X_train[:train_num], y_train[:train_num], batch_size)
        val_generator = self.__batch_generator(X_train[train_num:], y_train[train_num:], batch_size)

        history = self.custom_vgg_model.fit_generator(generator=train_generator, steps_per_epoch=steps, validation_data=val_generator, validation_steps=steps,
                                  epochs=epochs, callbacks=callbacks_list, workers=1)
        # history = self.custom_vgg_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size, callbacks=callbacks_list)

        return history

    def predict(self, image):
        image = self.processing(image)
        emotion = self.custom_vgg_model.predict(image)

        return emotion
