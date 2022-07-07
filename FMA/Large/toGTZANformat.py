#imports
import os
import shutil
import pandas as pd

#metadata for tracks.csv
filepath =  "../fma_metadata/tracks.csv"
data = pd.read_csv(filepath, index_col=0, header=[0, 1])
keep_cols = [('set', 'subset'),('track', 'genres_all'), ('track', 'genre_top'), ('track', 'genres')]
data = data[keep_cols]

#metadata for genres.csv
genrePath = "../fma_metadata/genres.csv"
data2 = pd.read_csv(genrePath)
keepCols = [('genre_id'), ('title')]
data2 = data2[keepCols]



genreList = []
idList = []
for i, j in data2.iterrows(): 
    genreList.append(j[1].split(" /", 1)[0])
    idList.append(j[0])
del data2

dictionary = dict(zip(idList, genreList))


X = []
Xstring = []
y = []
ok = []
count = 0
toRemove = []

for i, j in data.iterrows(): 
    #Medium and Small dataset have the field genre_top, which we will use here
    if(j[0] == 'medium' or j[0] == 'small'):
        Xstring.append("{0:0=6d}".format(i))
        y.append(j[2].split(" /", 1)[0])
    if(j[0] == 'large'):
    #Large dataset however does not contain genre_top. For that reason we take all genre (contained in another dictionnary) and keep only the first one
        a = j[1]
        a = a[1:-1]
        a = list(a.split(", "))
        a = a[0]
        if a == '':
            toRemove.append("{0:0=6d}".format(i))
            continue
        Xstring.append("{0:0=6d}".format(i))
        genre = dictionary[int(a)]
        #filters because some genre name contain forbidden chars
        if len(dictionary[int(a)].split(" /", 1)) > 1:
            genre = dictionary[int(a)].split(" /", 1)[0] + dictionary[int(a)].split(" /", 1)[1]
            
        if len(genre.split("/", 1)) > 1:
            genre = genre.split("/", 1)[0] + genre.split("/", 1)[1]
        
        if len(genre.split(": ", 1)) > 1:
            genre = genre.split(": ", 1)[0] + genre.split(": ", 1)[1]
        
        if len(genre.split(" :", 1)) > 1:
            genre = genre.split(" :", 1)[0] + genre.split(" :", 1)[1]
        
        if len(genre.split(":", 1)) > 1:
            genre = genre.split(":", 1)[0] + genre.split(":", 1)[1]
        y.append(genre)
    


rootDir= "./fma_large/"
currentDirectory = os.getcwd()
currentDirectory = currentDirectory.replace(os.sep, '/')



def findDir(filename):
    filename = str(filename)
    dir = filename[0:3]
    return dir

#remove files that dont have a genre
for i in range(0, len(toRemove)):
    file = toRemove[i]
    dir = findDir(file)
    print("file ", file, " is in dir ", dir)
    for _, _, files in os.walk(rootDir + dir):
        for f in files:
            if f == file + '.mp3':
                path = rootDir + dir + "/" + file + '.mp3'
                os.remove(path)
                print(file," removed")

#move the files to folders
for i in range(0, len(Xstring)):
    file = Xstring[i]
    genre = y[i]
    dir = findDir(file)
    for _, _, files in os.walk(rootDir + dir):
        for f in files:
            if f == file + '.mp3':
                source = rootDir + dir + "/" + file + '.mp3'
                os.makedirs(os.path.dirname(currentDirectory + "/genres/" + genre + "/"), exist_ok=True)
                destination = currentDirectory + "/genres/" + genre + "/" + file + ".mp3"
                shutil.move(source, destination)
                print(file , " moved to folder " + genre)
                break

#remove genres that are too small
for _, dirs, _ in os.walk('./genres/'): #List the directories in ./genres/
        for dir in dirs: #Go through each directory
            for _,_, files in os.walk('./genres/' + dir): #List the files in ./genres/subdir/
                if len(files) == 1:
                    shutil.rmtree('./genres/' + dir)
