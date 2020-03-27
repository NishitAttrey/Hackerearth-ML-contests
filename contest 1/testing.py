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
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

test=pd.read_csv('test.csv')
target={0:'Attire',1:'Food',2:'misc',3:'Decorationandsignage'}
testing_images=[]
model = tf.keras.models.load_model('model1.h5')
model.summary()
Class=[]
for i in tqdm(range(test.shape[0])):
	file = cv2.imread('Test Images/'+test['Image'][i])
	file = cv2.resize(file,(150,150))
	file = file.astype('float32')
	testing_images.append(file)

testing_images = np.array(testing_images)
labels = model.predict(testing_images)
label = [np.argmax(i) for i in labels]
class_label = [target[x] for x in label]
submission = pd.DataFrame({ 'Image': test.Image, 'Class': class_label })
submission.to_csv('submission.csv', index=False)