#imports
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from six.moves import cPickle as pickle
import audioread

#general variables
songLength = 660000
splitSize = 0.3
dataDirectory = './genres/'

#This function stores filename and label of each file of the dataset
#Then split it between training and testing set, and process both these sets
def preprocess_data():
    fileList = [] #List that will contain the names of every file with their location
    labelList = [] #List that will contain the genre of every music (integer)
    currentLabel = 0 #Counter that will increment each time we change folder
    for _, dirs, _ in os.walk(dataDirectory): #List the directories in ./genres/
        dirs.sort() #sort names to use alphabetic order
        for dir in dirs: #Go through each directory
            for _,_, files in os.walk(dataDirectory + dir): #List the files in subfolders
                for file in files: #Go through each file
                    fileList.append(dataDirectory + dir + '/' + file) #Add the path of the current file into fileList
                    labelList.append(currentLabel) #Add the genre to LabelList
            currentLabel = currentLabel + 1 # Increment the genre when we change directory
    
    
    #Split the data collected in fileList and labelList in a training set and a testing set
    #70% of files are kept for training, 30% for validation
    #No features were extracted yet, X is the path of the file, y is the genre [0:9]
    X_train, X_test, y_train, y_test = train_test_split(fileList,
                                                        labelList,
                                                        stratify = labelList,
                                                        test_size = splitSize
                                                        )
    
    #Split and convert Training and Testing Set
    processSet(X_train, y_train)
    processSet(X_test, y_test, isTest = True)


#This function a set as input (filename and labels)
#Then process them into melspectrogram and save into multiple pickle files (one for 50 songs)
def processSet(X, y, isTest = False):
    spectrograms = [] # List that contains the melspectrograms of 50 songs
    genres = [] #List that contains the label for 50 songs
    count = 0 #counter for file naming
    chunkSize = int(songLength*0.05) #chunk of size 5% of an entire song
    for i in range(0, len(X)):
        try:
            filename = X[i] #store the filename
            print("preprocessing data of " + filename)
            label = y[i] #store the label
            songData, sr = librosa.load(filename) #load the data of the song
            songData = songData[:songLength] #cut it to normalize it
            if len(songData) != songLength: #check if the song has the correct length, ignore if not
                continue
            for i in range(0, songLength, chunkSize):
                signal = np.array(songData[i:i+chunkSize])#load a chunk
                if(len(signal) != chunkSize): #check the size of the chunk
                    continue
                #add the spectrogram of the chunk to the array
                spectrograms.append(librosa.feature.melspectrogram(signal, n_fft=1024, hop_length=256, n_mels=128)[:,:,np.newaxis])
                #add the genre matching the spectrogram to the array
                genres.append(label)
                
            #flush when reaching 1000 spectrograms
            if(len(spectrograms) == 1000):
                savePickle(spectrograms, genres, isTest, count)
                spectrograms = [] #reset spectrograms and genres array
                genres = []
                count = count + 1 #update counter for file name
        except audioread.NoBackendError: #If audioread cant open file even with ffmpeg library, it is corrupted
            print(filename, " is corrupted, skipped. \nif this happens with every file, make sure you have install ffmpeg")


#Function that saves the pickle files
def savePickle(spec, genres, isTest, count):
    if isTest == False:
        pickle_file = 'train_' + str(count) + '.pickle'
        path = './train/'
    if isTest == True:
        pickle_file = 'test_' + str(count) + '.pickle'
        path = './test/'
    try:
        f = open(path + pickle_file, 'wb')
        save = {
            'X': spec,
            'y': genres,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print(pickle_file + "saved")
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise



# Create directories for training and testing files
try:
    # Create target Directory
    os.mkdir('./train/')
    print("train Directory Created ") 
except FileExistsError:
    print("Directory already exists")
    
try:
    # Create target Directory
    os.mkdir('./test/')
    print("test Directory Created ") 
except FileExistsError:
    print("Directory already exists")
    
    
#call preprocess_data
preprocess_data()

