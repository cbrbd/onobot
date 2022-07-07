import os
import shutil
import pandas as pd
import gc

#Gather the tracks meta data from tracks.csv
filepath = "../fma_metadata/tracks.csv" 
data = pd.read_csv(filepath, index_col=0, header=[0, 1])
keep_cols = [('set', 'subset'),('track', 'genres_all'), ('track', 'genre_top'), ('track', 'genres')]
data = data[keep_cols]

#arrays to keep filename and genre
Xstring = []
y = []

#go through the lines of tracks.csv
for i, j in data.iterrows(): 
    #Store name and genre for files in fma-small
    if(j[0] == 'small'):
        #X.append(i)
        Xstring.append("{0:0=6d}".format(i))
        y.append(j[2].split(" /", 1)[0])




#function that finds the dir of a file from its name
def findDir(filename):
    filename = str(filename)
    dir = filename[0:3]
    return dir

rootDir= "./fma_small/"
currentDirectory = os.getcwd()
currentDirectory = currentDirectory.replace(os.sep, '/')

#Go through each stored filename
for i in range(0, len(Xstring)):
    file = Xstring[i]
    genre = y[i]
    dir = findDir(file) #find dir
    for _, _, files in os.walk(rootDir + dir):
        for f in files:
            if f == file + '.mp3': #search for the file in the data
                source = rootDir + dir + "/" + file + '.mp3'
                os.makedirs(os.path.dirname(currentDirectory + "/genres/" + genre + "/"), exist_ok=True)
                destination = currentDirectory + "/genres/" + genre + "/" + file + ".mp3"
                shutil.move(source, destination) #move the data to GTZAN format
                print(file , ".mp3 moved to folder " + genre)
                break

del Xstring, data
gc.collect()