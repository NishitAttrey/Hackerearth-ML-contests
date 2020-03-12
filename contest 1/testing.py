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
	file = cv2.resize(file,(120,120))
	file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
	file = file/255.
	file=file.reshape((120,120,1))
	x = img_to_array(file)
	x=np.expand_dims(x,axis=0)
	image=np.vstack([x])
	belonging = model.predict_classes(image, batch_size=1)
	Class.append(target[belonging[0]])
	print(test['Image'][i],Class[i])

test['Class']=Class

test.set_index("Image",inplace=True)
print(test.head())
test.to_csv('test_result1.csv')