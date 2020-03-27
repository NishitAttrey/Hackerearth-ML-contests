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

di={'Attire':0,'Food':1,'misc':2,'Decorationandsignage':3}


train=pd.read_csv('train.csv')
training_images = []

for i in tqdm(range(train.shape[0])):
  img = cv2.imread('Train Images/'+train['Image'][i])
  img = cv2.resize(img,(150,150))
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
y = np.array(train.drop(['Image'],axis=1))
training_labels=np.zeros(len(training_images),dtype=int)

for i in range(len(y)):
  training_labels[i]=di[y[i][0]]

training_images=np.array(training_images)
print(training_images.shape)
print(training_labels.shape)
base_model=VGG16(include_top=False, weights='imagenet',input_shape=(150,150,3), pooling='avg')

model = tf.keras.Sequential()
model.add(base_model)

model.add(Dense(256,activation='relu'))
model.add(Dense(4,activation='softmax'))
base_model.trainable=False
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callbacks = [reduce_learning_rate]
model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(datagen.flow(training_images, to_categorical(training_labels,4), batch_size=32),
                    epochs=20,callbacks=callbacks)
model.save("model1.h5")
print("Saved model to disk")