import keras
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

di={'Attire':0,'Food':1,'misc':2,'Decorationandsignage':3}

class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('acc')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training=True


train=pd.read_csv('train.csv')
training_images = []

for i in tqdm(range(train.shape[0])):
	img = cv2.imread('Train Images/'+train['Image'][i],cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img,(120,120))
	img = image.img_to_array(img)
	img = img/255
	training_images.append(img)

X = np.array(training_images)
y = np.array(train.drop(['Image'],axis=1))
training_labels=np.zeros(len(training_images),dtype=int)

for i in range(len(y)):
	training_labels[i]=di[y[i][0]]

callbacks=mycallback()

training_images=X.reshape(len(X),120,120,1)

model=tf.keras.Sequential([tf.keras.layers.Conv2D(128,(2,2),activation='relu',input_shape=(120,120,1)),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128,activation='relu'),
	tf.keras.layers.Dense(4,activation='softmax')
    ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

model.fit(X,training_labels,epochs=18,callbacks=[callbacks])
model.save("model1.h5")
print("Saved model to disk")