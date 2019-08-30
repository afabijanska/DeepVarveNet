# DeepVarveNet

This crepository contains source code of a DeepVarveNet - a convolutional neural network for laminae detection in images of varved sediments.

# Running the code

## Prerequisites

Python 3.6, Tensorflow, Keras

## Repository content

<ul>
  <li> <b>configuration.txt</b> - file to be edited; contains data paths and train/test setings;   
  <li> <b>ExtractMarkers.py</b> - script for extracting manual varve delineations provided by an expert; it is assumed that delineations are in pure blue (R=0, G=0, B=255);
  <li> <b>DivideTestTrain.py</b> - script for dividing data into training and testing set; 
  <li> <b>GlaciersGenerateData.py</b> - script for extracting train patches and the corresponding labels from the images from the train set; it is assumed that an equal numner of patches is extracted from each train image;
  <li> <b>GlaciersTrainCNNpix.py</b> - script for training the DeepVarveNet with the train patches and the corresponding labels;
  <li> <b>GlaciersPredict.py</b> - script for finding varves in test images with the use of trained DeepVarveNet;
  <li> <b>models.py</b> - file that defines architecture of a DeepVarveNet;
  <li> <b>GlaciersHelpers.py</b> - some helper functions;
</ul>

# Contact

Anna Fabijańska<br>
Institute of Applied Computer Science<br>
Lodz University of Technology<br>
e-mail: anna.fabijanska@p.lodz.pl<br> 
WWW: http://an-fab.kis.p.lodz.pl
