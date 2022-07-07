import os
import shutil
import pandas as pd
import numpy as np

#path for the metadata tracks.csv
filepath = "../fma_metadata/tracks.csv"

#read metadata and keep interesting column
data = pd.read_csv(filepath, index_col=0, header=[0, 1])
keep_cols = [('set', 'subset'),('track', 'genres_all'), ('track', 'genre_top'), ('track', 'genres')]
data = data[keep_cols]

#arrays to store file names and labels
Xstring = []
y = []


for i, j in data.iterrows(): 
    #search for files from fma-small and fma-medium
    if(j[0] == 'small' or j[0] == 'medium'):
        Xstring.append("{0:0=6d}".format(i))
        y.append(j[2].split(" /", 1)[0]) #get rid of the "/" in the "Old Time / historic" that causes problem


#path to data
rootDir= "./fma_medium/"
currentDirectory = os.getcwd()
currentDirectory = currentDirectory.replace(os.sep, '/')


#function to find a dir from filename
def findDir(filename):
    filename = str(filename)
    dir = filename[0:3]
    return dir



for i in range(0, len(Xstring)):
    file = Xstring[i]
    genre = y[i]
    dir = findDir(file)
    for _, _, files in os.walk(rootDir + dir):
        for f in files:
            if f == file + '.mp3': #search for the file
                source = rootDir + dir + "/" + file + '.mp3'
                os.makedirs(os.path.dirname(currentDirectory + "/genres/" + genre + "/"), exist_ok=True)
                destination = currentDirectory + "/genres/" + genre + "/" + file + ".mp3"
                shutil.move(source, destination) #move file to the matching folder
                print(file + ".mp3 moved to folder " + genre)
                break
