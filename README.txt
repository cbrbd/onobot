How to use ?

This project is using two different datasets, GTZAN and FMA. To test pre-processing and training, I would advice 
trying GTZAN, as the calculation time is much faster. The computer used to develop and test these programs is:
-GPU: Nvidia 1050Ti
-CPU: intel Core i7-8750H
-RAM: 16 GB 2667MHz

To run the different files, you can use anaconda with Python 3 and create a new environnement for this project. You can then use Spyder to run the files 

Here are the libraries you will need:

1. tensorflow	conda install tensorflow

or you can use NVIDIA cuDNN to accelerate calculation with gpu usage https://github.com/antoniosehk/keras-tensorflow-windows-installation

2. librosa	pip install librosa
		conda install pip

3. ffmpeg 	conda install -c conda-forge ffmpeg
(backend to read mp3)	

3. keras 	conda install keras-gpu

4. matplotlib	conda install matplotlib

5. pandas 	conda install pandas



it is worth to mention that the file "evaluate.py" will not be able to evaluate the accuracy of the pretrained model given, as the random state of the distribution of the songs will not be the same


1) GTZAN

- To test pre-processing and training, you must download the GTZAN dataset from this URL http://marsyas.info/downloads/datasets.html. In the GTZAN project folder, replace the empty "genres" folder with the downloaded one. Then run "preprocess.py". Once finished, run "TrainCNN.py". The number of epochs can be changed at line 23. If you want to plot the confusion matrix and the grouped accuracy in Spyder, run "evaluate.py".


- To test prediction, you do not need to download the dataset. Run "predict.py" and wait until a window opens to select a music file (wav, wma and mp3 are working well). By default, the pre-trained model will be used. If you want to use another trained model, change line 14 to useDefault = True. This will load 'trainedCNN.h5'.


2) FMA-small

- To test preprocessing and training, download the fma_small.zip and fma_metadata.zip from this URL https://github.com/mdeff/fma. In the folder "FMA", change the empty folder "fma_metadata" with the one downloaded. In the folder "Small", change the empty folder "fma_small" with the one downloaded. 
Next, run "toGTZANformat.py", then "preprocess.py", then "TrainCNN.py". To evaluate the model, run "evaluate.py"

- To test prediction, you do not need to download the dataset. Run "predict.py" and wait until a window opens to select a music file (wav, wma and mp3 are working well). By default, the pre-trained model will be used. If you want to use another trained model, change line 14 to useDefault = True. This will load 'trainedCNN.h5'.


3) FMA-medium

- To test preprocessing and training, download the fma_medium.zip and fma_metadata.zip from this URL https://github.com/mdeff/fma. In the folder "FMA", change the empty folder "fma_metadata" with the one downloaded. In the folder "Medium", change the empty folder "fma_medium" with the one downloaded. 
Next, run "toGTZANformat.py", then "preprocess.py", then "TrainCNN.py". To evaluate the model, run "evaluate.py"

- To test prediction, you do not need to download the dataset. Run "predict.py" and wait until a window opens to select a music file (wav, wma and mp3 are working well). By default, the pre-trained model will be used. If you want to use another trained model, change line 14 to useDefault = True. This will load 'trainedCNN.h5'.

4) FMA-Large

- To test preprocessing and training, download the fma_large.zip and fma_metadata.zip from this URL https://github.com/mdeff/fma. In the folder "FMA", change the empty folder "fma_metadata" with the one downloaded. In the folder "Large", change the empty folder "fma_large" with the one downloaded. 
Next, run "toGTZANformat.py", then "preprocess.py", then "TrainCNN.py". To evaluate the model, run "evaluate.py"

- To test prediction, you do not need to download the dataset. The default model is trained on 21 genres. Run "predict.py" and wait until a window opens to select a music file (wav, wma and mp3 are working well). By default, the pre-trained model will be used. If you want to use another trained model, change line 14 to useDefault = True. This will load 'trainedCNN.h5'.