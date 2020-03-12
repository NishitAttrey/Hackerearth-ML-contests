import keras
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2      
di={'Attire':0,
	'Food':1,
	'misc':2,
	'Decorationandsignage':3}

id = {
    0: 'Food',
    1: 'Attire',
    2: 'Decorationandsignage',
    3: 'misc'
}
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Class'] = train['Class'].map(di).astype(np.int8)
print(train.head())

train_path = 'Train Images/'
test_path = 'Test Images/'
train_images,train_labels =[], []

for i in range(len(train.Image)):
  train_image = cv2.imread(train_path + str(train.Image[i]))
  train_image = cv2.resize(train_image, (224, 224))
  train_images.append(train_image)
  train_labels.append(train.Class[i])

test_images =[]

for i in range(len(test.Image)):
  test_image = cv2.imread(test_path + str(test.Image[i]))
  test_image = cv2.resize(test_image, (224, 224))
  test_images.append(test_image)

train_images = np.array(train_images)
test_images = np.array(test_images)

X_train, X_test, y_train, y_test = train_test_split(train_images, to_categorical(train_labels), test_size=0.2, random_state=42)

models = ResNet50(
		weights='imagenet',
		include_top = False,
		input_shape  = (224,224,3),
		pooling = 'avg'
	)

models.trainable = False

model = Sequential([
		models,
		Dropout(0.2),
		Dense(4,activation='softmax')
	])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

batch_size = 32
epochs = 5


datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), validation_data= (X_test, y_test),
                    steps_per_epoch=len(X_train) / batch_size, epochs=epochs)

labels = model.predict(test_images)
label = [np.argmax(i) for i in labels]
class_label = [inverse_class_map[x] for x in label]
submission = pd.DataFrame({ 'Image': test_df.Image, 'Class': class_label })
submission.head()
submission.to_csv('sub.csv', index=False)