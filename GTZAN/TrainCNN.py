#General imports
import numpy as np
import os
from six.moves import cPickle as pickle

#Data representation imports
import matplotlib.pyplot as plt

#tensorflow imports
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Convolution2D

#general variables
train_size = 1000
test_size = 1000
batch_size = 512
epochs = 25

#retrieve the number of pickle file for training
for _, _, files in os.walk('./train/'): #List the directories in ./genres/
    numberOfTrainFile = len(files)
    pickleData = pickle.load(open('./train/' + files[0], "rb"))
    X = np.array(pickleData['X'])
    shape = X.shape
    del pickleData, X
         
#retrieve the number of pickle file for validation
for _, _, files in os.walk('./test'): #List the directories in ./genres/
    numberOfTestFile = len(files)


#create a dictionnary of all the genres based on the folder
genres = []
for _, dirs, _ in os.walk('./genres/'): #List the directories in ./genres/
        for dir in dirs: #Go through each directory
            genres.append(dir)
genres.sort()
classes = len(genres)

#data generator for training  
def raw_generator(directory, batch_size):
    counter = 0
    while 1:      
        for _,_, files in os.walk(directory):
            fname = directory + '/' + files[counter]
            counter = (counter + 1) % len(files)
            pickleData = pickle.load(open(fname, "rb"))
            
            X_train = np.array(pickleData['X'])
            y_train = to_categorical(pickleData['y'], num_classes = classes)
            for cbatch in range(0, X_train.shape[0], batch_size):
                yield (X_train[cbatch:(cbatch + batch_size),:,:], y_train[cbatch:(cbatch + batch_size)])
        

#data generator for validation
def raw_test_gen(directory, batch_size):
    counter = 0
    while 1:      
        for _,_, files in os.walk(directory):
            fname = directory + '/' + files[counter]
            counter = (counter + 1) % len(files)
            pickleData = pickle.load(open(fname, "rb"))
            
            X_train = np.array(pickleData['X'])
            y_train = to_categorical(pickleData['y'], num_classes = classes)
            for cbatch in range(0, X_train.shape[0], batch_size):
                yield (X_train[cbatch:(cbatch + batch_size),:,:], y_train[cbatch:(cbatch + batch_size)])

#function that plots accuracy and loss curves from history of training
def plot_acc_and_loss(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    #plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()




trainDir = './train'
testDir = './test'

train_gen = raw_generator(trainDir, batch_size)
test_gen = raw_test_gen(testDir, batch_size)



model = Sequential()

model.add(Convolution2D(16, (3,3), activation='relu', input_shape=shape[1:4], data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))
model.add(Dropout(0.25))

model.add(Convolution2D(32, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))
model.add(Dropout(0.25))

model.add(Convolution2D(128, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))
model.add(Dropout(0.25))


model.add(Convolution2D(256, (3,3)))
model.add(MaxPooling2D(pool_size=(2,1), data_format='channels_last'))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(classes, activation='softmax'))

print(model.summary())

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])




#fit the model on training data and validation data
history = model.fit(
    train_gen,
    steps_per_epoch= (numberOfTrainFile*train_size)//batch_size,
    validation_data = test_gen,
    validation_steps = (numberOfTestFile*test_size)//batch_size,
    epochs=epochs,
    verbose=1,
    )

#create the folder for saving the model
try:
    # Create target Directory
    os.mkdir('./models/')
    print("Directory Created ") 
except FileExistsError:
    print("Directory already exists")

#save the model
model.save('./models/trainedCNN.h5')

#Accuracy and Loss
plot_acc_and_loss(history)




