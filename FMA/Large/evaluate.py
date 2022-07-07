import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from six.moves import cPickle as pickle
from six.moves import range
import gc

import tensorflow as tf

np.seterr(divide='ignore', invalid='ignore')

#general parameters
model = 'trainedCNN.h5'
train_size = 1000
test_size = 1000
batch_size = 512


#go through training pickle files to count how many files there are, and to collect the shape of the data
for _, _, files in os.walk('./train/'): 
    numberOfTrainFile = len(files)
    pickleData = pickle.load(open('./train/' + files[0], "rb"))
    X = np.array(pickleData['X'])
    shape = X.shape
    del pickleData, X
         
    
#retrieve the number of pickle file for validation
for _, _, files in os.walk('./test'): #List the directories in ./genres/
    numberOfTestFile = len(files)

#build the list of genres from the genres folder
genres = []
for _, dirs, _ in os.walk('./genres/'): #List the directories in ./genres/
        for dir in dirs: #Go through each directory
            genres.append(dir)    
genres.sort()

classes = len(genres)
dictionary = dict(zip(genres, list(range(0, classes))))


#Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues, png_output=None, show=True):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title='Normalized confusion matrix'
        else:
            title='Confusion matrix'

        # Calculate chart area size
        leftmargin = 0.5 # inches
        rightmargin = 0.5 # inches
        categorysize = 0.5 # inches
        figwidth = leftmargin + rightmargin + (len(classes) * categorysize)           

        f = plt.figure(figsize=(figwidth, figwidth))

        # Create an axes instance and ajust the subplot size
        ax = f.add_subplot(111)
        ax.set_aspect(1)
        f.subplots_adjust(left=leftmargin/figwidth, right=1-rightmargin/figwidth, top=0.94, bottom=0.1)

        res = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.title(title)
        plt.colorbar(res)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if png_output is not None:
            os.makedirs(png_output, exist_ok=True)
            f.savefig(os.path.join(png_output,'confusion_matrix.png'), bbox_inches='tight')

        if show:
            plt.show()
            plt.close(f)
        else:
            plt.close(f)
       

#function to get all the predicted labels and true labels of the validation files
def get_orig(directory, model):
    preds = []
    y_orig = []
    for _,_, files in os.walk(directory):
        for file in files:
            fname = directory + '/' + file
            pickleData = pickle.load(open(fname, "rb"))
            X_test = np.array(pickleData['X'])
            y_test = pickleData['y']
            pred = np.argmax(model.predict(X_test), axis = 1)
            
            for p in pred:
                preds.append(p)
            for y in y_test:
                y_orig.append(y)
    return preds, y_orig


#function to get the most represented number in a list
def song_majority(scores):
    unique, counts = np.unique(scores, return_counts=True)
    higherGuess = np.argmax(counts)
    return unique[higherGuess]


#function to evaluate the accuracy on 30 second segments
def evaluate_majority_acc(directory, model, numberOfTestFile, predictionSplitted, label):
    numberOfSong = numberOfTestFile * 50 #50 songs per file
    predictionGrouped = np.split(np.array(predictionSplitted), numberOfSong) #put segments together to reassemble the songs
    del predictionSplitted
    for i in range(0,len(predictionGrouped)):
        predictionGrouped[i] = song_majority(predictionGrouped[i]) #get the most predicted genre in the song
    
    label = np.split(np.array(label), numberOfSong) #same for the labels
    for i in range(0,len(label)):
        label[i] = song_majority(label[i])
        
    #evaluate percent of good prediction with groupped
    count = 0
    for x, y in zip(predictionGrouped, label):
        if x == y:
            count = count + 1
    groupAccuracy = (count/numberOfSong)*100
    print("Accuracy on 30sec song with majority vote : ","%.2f" % groupAccuracy, "%")
    del predictionGrouped, label


#load the model
model = tf.keras.models.load_model('./models/' + model)


#Gather data for confusion matrix and majority vote
preds, y_orig = get_orig('./test', model)

#plot confusion matrix
cm = confusion_matrix(preds, y_orig)
keys = OrderedDict(sorted(dictionary.items(), key=lambda t: t[1])).keys()
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, keys, normalize=True)



#print grouped accuracy
evaluate_majority_acc('./test', model, numberOfTestFile, preds, y_orig)

del preds, y_orig, cm, keys
gc.collect()