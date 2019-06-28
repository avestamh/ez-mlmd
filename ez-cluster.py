#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import load_img
from keras import backend
from sklearn.cluster import KMeans
import numpy as np

# load images listed in $inp_name and store in array
def load_img_arr(inp_name):
    with open(inp_name, 'r') as inp_file:
        img_list = [each_name.strip() for each_name in inp_file]
    img_arr = np.stack([load_img(each_img) for each_img in img_list])
    return img_arr.astype('float32')/255., img_list

#---------- 0. User input ----------
n_clusters = 3 # determined from 2D histogram of all data
inp_name = "input-files.dat" # files containing the path of all training data
input_shape = (256, 256, 3)  # img_width, img_height, channels

#---------- 1. Train the model ----------
X, img_list = load_img_arr(inp_name)
model = Sequential()
# encoder part : (conv + relu + maxpooling) x 3
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(8, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
# decoder part : (conv + relu + upsampling) x 3
model.add(Conv2D(8, kernel_size=(3,3), activation='relu', padding='same'))
model.add(UpSampling2D(size=(2,2)))
model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
model.add(UpSampling2D(size=(2,2)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(UpSampling2D(size=(2,2)))
model.add(Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same'))
# compile and train the model
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(X, X, epochs=10, batch_size=5, shuffle=True, verbose=1)

#---------- 2. Retrieve encoded image and classify pathways ----------
get_encoded_layer = backend.function([model.layers[0].input],[model.layers[5].output])
encoded_layer = get_encoded_layer([X])[0]
X_encoded = encoded_layer.reshape(encoded_layer.shape[0], -1)
km = KMeans(n_clusters)
km.fit(X_encoded)

#---------- 3. Print percentage of each path and corresponding example image ----------
X_clustered = km.labels_
N = float(len(X_clustered))
paths, counts = np.unique(X_clustered, return_counts=True)
print "---Output---"
for each_path, each_count in zip(paths, counts):
    idx = np.where(X_clustered==each_path)[0][0]
    print "path%d (%.2f) %s"%(each_path+1, each_count/N, img_list[idx])