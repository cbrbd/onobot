import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

from tkinter.filedialog import askopenfilename
from tkinter import *

useDefault = True


#model name
if useDefault == True:
    modelName = 'SmallPretrainedCNN.h5'
    genres = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]
    genres.sort()
else:
    modelName = 'trainedCNN.h5'
    genres = []
    for _, dirs, _ in os.walk('./genres/'): #List the directories in ./genres/
            for dir in dirs: #Go through each directory
                genres.append(dir)
    if len(genres) == 0:
        genres = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]
    genres.sort()
    
print(genres)

#preocess the song to predict
def processSong(songData):
    songLength = songData.shape[0]
    chunkSize = 33000
    spectrograms = []
    for i in range(0, songLength, chunkSize):
            signal = np.array(songData[i:i+chunkSize])#load a chunk
            if(len(signal) != chunkSize): #check the size of the chunk
                continue
            #add the spectrogram of the chunk to the array
            spectrograms.append(librosa.feature.melspectrogram(signal, n_fft=1024, hop_length=256, n_mels=128)[:,:,np.newaxis])
            #add the genre matching the spectrogram to the array
    spectrograms = np.array(spectrograms)
    return spectrograms


#Function that loads the model and predict a song
def predictGenre():
    #Open a window to ask which music to guess
    root = Tk()
    root.attributes('-topmost', True) # note - before topmost
    root.withdraw()
    filename = askopenfilename()
    root.destroy()
    
    
    print("Predicting genre for ", filename)
    signal, sr = librosa.load(filename) #loading the file
    melspec = processSong(signal) #process the file into melspec of shape (128, 129, 1)
    CNN = tf.keras.models.load_model('./models/' + modelName) #load the model
    prediction = CNN.predict_classes(melspec) #predict the classes
    print("\nsong is composed of " + str(len(prediction)) + " segments")
    
    #retrieve genre and count for each genre
    unique, counts = np.unique(prediction, return_counts=True)
    labels = genres
    total = 0
    higherGuess = np.argmax(counts)
    
    #calculate percentage
    for c in counts:
        total = total + c
    
    for genre, count in zip(unique, counts):
        stat = (count/total) * 100
        print(labels[genre], " : ", "%.2f" % stat , "%")
    
    print("\nmost probable : ", labels[unique[higherGuess]])
    
    #plot histogram
    timeAxis = []
    for i in range(0, len(prediction)):
        timeAxis.append(i*1.5)

    yAxis = []
    for i in range(0, len(prediction)):
        genre = labels[prediction[i]]
        yAxis.append(labels[prediction[i]])

    plt.plot(timeAxis, yAxis, ls = "None", marker=".", markersize = 5)
    plt.show()



predictGenre()

