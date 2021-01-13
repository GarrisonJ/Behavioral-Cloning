import glob, os
import csv
import cv2
import numpy as np
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, Cropping2D, Dropout

path_to_training_data = '../../../opt/training/'

images = []
measurements = []

for folder in glob.iglob(path_to_training_data + "**"):
    lines = []
    with open(folder + '/driving_log.csv') as cvsfile:
        reader = csv.reader(cvsfile)
        for line in reader:
            lines.append(line)
    

    for line in lines:        
        steering = float(line[3])
        correction = 0.2
        steering_left = steering + correction
        steering_right = steering - correction        
        
        img_center_source_path = line[0]
        file_name = img_center_source_path.split('\\')[-1]
        img_center = ndimage.imread(folder + '/IMG/' + file_name)
        
        img_left_source_path = line[1]
        file_name = img_left_source_path.split('\\')[-1]
        img_left = ndimage.imread(folder + '/IMG/' + file_name)
        
        img_right_source_path = line[2]
        file_name = img_right_source_path.split('\\')[-1]
        img_right = ndimage.imread(folder + '/IMG/' + file_name)
        
        images.extend([img_center, img_left, img_right])
        measurements.extend([steering, steering_left, steering_right])


X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(12, (5,5), strides=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(60, (3,3), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(72, (3,3), activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
          
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
model.save('model.h5')
