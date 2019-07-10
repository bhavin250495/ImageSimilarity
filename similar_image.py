import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
from keras.applications.mobilenet import preprocess_input
import os
print("TensorFlow version is ", tf.__version__)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Getting pretrained Model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
base_model.summary()



# Feature Vector

feature_list = []

path = 'test/'

for idx, dirname,f in os.walk(path):
    for i in f:


        img_path = 'test/' + i
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        feature = base_model.predict(img_data)
        feature_np = np.array(feature)
        feature_list.append(feature_np.flatten())


feature_list = np.array(feature_list)
test_list = feature_list[:5]
from sklearn.metrics.pairwise import cosine_similarity
simi = cosine_similarity(feature_list)

y1 = np.zeros(800,dtype=int)
y2 = np.ones(55,dtype=int)
y = np.concatenate((y1, y2), axis=0)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(feature_list,y)

img_path = 'data/accordion/image_0001.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)
feature = base_model.predict(img_data)
feature_np = np.array(feature)
d = model.predict(feature_np.flatten().reshape(-1, 1))






