import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from six.moves import cPickle as pickle


songLength = 660000
splitSize = 0.3
dataDirectory = './genres/'

#This function reads the dataset and splits it between training set and testing set
#Train and Test sets are then processed by function processSet()
def preprocess_data():
    fileList = [] #List that will contain the names of every file with their location
    labelList = [] #List that will contain the genre of every music (integer between 0 and 9)
    currentLabel = 0 #Counter that will increment each time we change genre
    for _, dirs, _ in os.walk(dataDirectory): #List the directories in ./genres/
        dirs.sort()
        for dir in dirs: #Go through each directory
            for _,_, files in os.walk(dataDirectory + dir): #List the files in ./genres/subdir/
                for file in files: #Go through each file
                    fileList.append(dataDirectory + dir + '/' + file) #Add the path of the current file into fileList
                    labelList.append(currentLabel) #Add the genre to LabelList
            currentLabel = currentLabel + 1 # Increment the genre when we change directory
    
    
    #Split the data collected in fileList and labelList in a training set and a testing set
    #700 files are kept for the training, 300 for testing
    #No features were extracted yet, X is the path of the file, y is the genre [0:9]
    X_train, X_test, y_train, y_test = train_test_split(fileList,
                                                        labelList,
                                                        stratify = labelList,
                                                        test_size = splitSize
                                                        )
    
    #Split and convert Training and Testing Set
    processSet(X_train, y_train)
    processSet(X_test, y_test, isTest = True)



def processSet(X, y, isTest = False):
    spectrograms = [] # List that contains the melspectrograms of 50 songs
    genres = [] #List that contains the label for 50 songs
    count = 0 #counter for file naming
    chunkSize = int(songLength*0.05) #chunk of size 5% of an entire song
    for i in range(0, len(X)):
        filename = X[i] #store the filename
        print(filename)
        label = y[i] #store the label
        songData, sr = librosa.load(filename) #load the song
        songData = songData[:songLength] #cut it to normalize it
        
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
            spectrograms = []
            genres = []
            count = count +1


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
        print(pickle_file + " saved")
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise



# Read the data
try:
    # Create target Directory
    os.mkdir('./train/')
    print("Directory Created ") 
except FileExistsError:
    print("Directory already exists")
    
try:
    # Create target Directory
    os.mkdir('./test/')
    print("Directory Created ") 
except FileExistsError:
    print("Directory already exists")
    
preprocess_data()

