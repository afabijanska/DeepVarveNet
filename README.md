# DeepVarveNet

This crepository contains source code of the DeepVarveNet - a convolutional neural network for laminae detection in images of varved sediments.

# Running the code

## Prerequisites

Python 3.6, Tensorflow, Keras, Anaconda3

## Data organization

Organize your data as below. For training, keep the filenames consistent (an original image and the corresponding ground truth should be files of the same name saved in diifferent locations).

<pre><code>
├───project_dir<br>
    └───allData<br>                 # all image data available in the project (contains separate subdir for each core)
    |    └───DataDir1<br>           # images of the first sendiment core
    |    |   └───bw<br>             # binary images (varve masks)
    |    |   |   File1.png <br>
    |    |   |   File2.png <br>
    |    |   └───gt<br>             # mannual delineations by an expert
    |    |   |   File1.png <br>
    |    |   |   File2.png <br>
    |    |   └───src<br>            # original images of varves
    |    |       File1.png <br>
    |    |       File2.png <br>
    |    └───DataDir2<br>           # images of the second sendiment core
    |        └───bw<br>             # binary images (varve masks)
    |        |   File3.png <br>
    |        |   File4.png <br>
    |        └───gt<br>             # mannual delineations by an expert
    |        |   File3.png <br>
    |        |   File4.png <br>
    |        └───src<br>            # original images of varves
    |            File3.png <br>
    |            File4.png <br>
    └───train<br>                   # train data 
    |    └───bw<br>
    |    |   DataDir1_File1.png <br>    #filenames are generated automatically 
    |    |   DataDir2_File3.png <br>
    |    └───src<br>
    |        DataDir1_File1.png <br>
    |        DataDir2_File3.png <br>
    └───test<br>                    # test data  
         └───bw<br>
         |   DataDir1_File2.png <br>
         |   DataDir2_File4.png <br>
         └───final<br>              # predicted varves overlied on original images
         |   DataDir1_File2.png <br>
         |   DataDir2_File4.png <br>
         └───preds<br>              # varve probability maps 
         |   DataDir1_File2.png <br>
         |   DataDir2_File4.png <br>
         └───src<br>
             DataDir1_File2.png <br>
             DataDir2_File4.png <br>
        
</code></pre>

# Repository content

<ul>
  <li> <b>configuration.txt</b><br> -- file to be edited; <br> -- contains data paths and train/test setings;   
  <li> <b>ExtractMarkers.py</b><br> -- script for extracting manual varve delineations provided by an expert; <br> -- to be run 1st; <br> -- expects that for each sediment core subdir ./src contains original images while subdir ./gt contains manual delineations by an expert<br> -- it is assumed that delineations are provided in pure blue (R=0, G=0, B=255) as markings on original images; <br> -- for each core the extracted varve binary masks are saved in subdir ./bw;  
  <li> <b>DivideTestTrain.py</b><br> -- script for dividing data into training and testing set; <br> -- to be run 2nd; <br> -- copies every second image to ./train and ./test dir respectively;
  <li> <b>GlaciersGenerateData.py</b><br> -- script for extracting training patches and the corresponding labels from the train images; <br> -- to be run 3rd; <br> -- an equal number of patches is extracted from each train image; 
  <li> <b>GlaciersTrainCNNpix.py</b><br> -- script for training the DeepVarveNet with the train patches and the corresponding labels; <br> -- to be run 4th;
  <li> <b>GlaciersPredict.py</b><br> -- script for finding varves in test images with the use of trained DeepVarveNet; <br> -- to be run 5th; <br> -- varve likelihood maps are saved in subdir ./preds; <br> -- final varves overlied on original data are saved in subdir ./final;
  <li> <b>models.py</b><br> -- file that defines architecture of the DeepVarveNet;
  <li> <b>GlaciersHelpers.py</b><br> -- some helper functions;
  <li> <b>GlaciersEvaluate.py</b><br> -- script to compare results of automatic varves detection with ground truths provided by an expert;
</ul>

# Contact

<b>Anna Fabijańska</b><br>
Institute of Applied Computer Science<br>
Lodz University of Technology<br>
e-mail: anna.fabijanska@p.lodz.pl<br> 
WWW: http://an-fab.kis.p.lodz.pl
