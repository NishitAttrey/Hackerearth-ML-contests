import keras
import tensorflow as tf 
from tensorflow.keras.models import Model,Sequential
from keras.preprocessing import image
from keras.preprocessing.image import  ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau

di={'angry':0,'happy':1,'sad':2,'surprised':3,'Unknown':4}

train=pd.read_csv('Train.csv')
training_images = []

for i in tqdm(range(train.shape[0])):
  img = cv2.imread('train_frames/'+train['Frame_ID'][i])
  img = cv2.resize(img,(144,144))
  img = img/255.
  img = img.astype('float32')
  training_images.append(img)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(training_images)
y = np.array(train.drop(['Frame_ID'],axis=1))
training_labels=np.zeros(len(training_images),dtype=int)

for i in range(len(y)):
  training_labels[i]=di[y[i][0]]

training_images=np.array(training_images)
print(training_images.shape)
print(training_labels.shape)
base_model=VGG16(include_top=False, weights='imagenet',input_shape=(144,144,3), pooling='avg')

model = tf.keras.Sequential()
model.add(base_model)

model.add(Dense(256,activation='relu'))
model.add(Dense(5,activation='softmax'))
base_model.trainable=False
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callbacks = [reduce_learning_rate]
model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(datagen.flow(training_images, to_categorical(training_labels,5), batch_size=32),
                    epochs=50,callbacks=callbacks)
model.save("model2.h5")
print("Saved model to disk")