# DeepVarveNet

This crepository contains source code of a DeepVarveNet - a convolutional neural network for laminae detection in images of varved sediments.

# Running the code

## Prerequisites

Python 3.6, Tensorflow, Keras

## Repository content

<ul>
  <li> <b>configuration.txt</b><br> -- file to be edited; <br> -- contains data paths and train/test setings;   
  <li> <b>ExtractMarkers.py</b><br> script for extracting manual varve delineations provided by an expert; it is assumed that delineations are provided in pure blue (R=0, G=0, B=255) as markings on original images (to be run 1st);
  <li> <b>DivideTestTrain.py</b><br> script for dividing data into training and testing set (to be run 2nd); 
  <li> <b>GlaciersGenerateData.py</b><br> script for extracting train patches and the corresponding labels from the images from the train set; an equal number of patches is extracted from each train image (to be run 3rd);
  <li> <b>GlaciersTrainCNNpix.py</b><br> script for training the DeepVarveNet with the train patches and the corresponding labels (to be run 4th);
  <li> <b>GlaciersPredict.py</b><br> script for finding varves in test images with the use of trained DeepVarveNet (to be run 5th);
  <li> <b>models.py</b><br> file that defines architecture of a DeepVarveNet;
  <li> <b>GlaciersHelpers.py</b><br> some helper functions;
</ul>

# Contact

<b>Anna Fabijańska</b><br>
Institute of Applied Computer Science<br>
Lodz University of Technology<br>
e-mail: anna.fabijanska@p.lodz.pl<br> 
WWW: http://an-fab.kis.p.lodz.pl
