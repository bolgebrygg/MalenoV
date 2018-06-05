# MalenoV_nD
<h1> Tool for training &amp;  classifying 3D (4D, nD) SEGY seismic facies using deep neural networks</h1>

<h3>More modern versions of this tool including fully convolutional encoder- decoder networks can be found on this continuation of the project https://github.com/crild/facies_net </h3>

•	MalenoV reads standard 3D SEGY seismic and performs a 3D neural network architecture of choice on a given set of classification data points (facies annotation /supervision).  It then uses the learned weights and filters of the neural network to classify seismic at any other location in the seismic cube into the facies classes that have been previously been defined by the user. Finally the facies classification is written out as a SEGY cube with the same dimensions as the input cube.

The tool reads can handle n seismic input cubes (offest stacks, 4D data, attributes, etc) and n number of facies training datasets. The more input seismic cubes are trained / classified  simultaneously the more memory is needed and the slower training /classification will be (linear scaling).

•	MalenoV was created as part of the Summer project of Charles Rutherford Ildstad in ConocoPhillips in July 2017.

•	The tool was inspired by the work of Anders U. Waldeland who showed at that he was successfully classifying the seismic facies of salt using 3D convolutional networks (http://earthdoc.eage.org/publication/publicationdetails/?publication=88635). 

•	Currently a 5 layer basic 3D convolutional network is implemented but this can be changed by the users at liberty. 

•	This tool is essentially an I/O function for machine learning with seismic. Better neural architectures for AVO, rock property prediction, fault classifications are to be implemented. (Think Unet resnet or GAN).

•	The tool is public with a GNU Lesser General Public License v3.0

•	The tool  has been updated to handle multiple input volumes (offest stacks, 4D seismic) for better classification results and more fun

<h3>The User Manual, seismic data, training data and set up scripts for the tool can be found here</h3>
https://goo.gl/wb145Z





<h3>Seismic training data for testing the tool:</h3>

•	We decided to make available the Poseidon (3500km2) seismic dataset acquired for ConocoPhillips Australia including Near, Mid, Far stacks, well data and velocity data

•	The seismic data is available here: https://goo.gl/wb145Z 
<b> BEAWRE one 32 bit SEGY File is 100 GB of data</b>

•	There is also inline, xline, z, training data for fault identification on the Poseidon survey

•	The Dutch government F3 seismic dataset can also be downloaded from the same location. 
<b>This data is only 1 GB</b>

•	Training data locations for multi facies prediction, faults and steep dips is provided
•	Trained neural network models for steep dips and multi facies can be assessed
. 
.

<b>MalenoV stands for MAchine LEarNing Of Voxels</b>

<b>nD stands for unlimited input dimensions</b>
 
 
  
 .
 .
 <h3>Improvement ideas:</h3>
 
 <b>Priority number 1 </b>is to improve the classification speed.

One easy  solution for this would be to do the classification only at a user defined spacing. Say every second inline and every third crossline and every other z sample. Once the classification is done a segy cube needs to be written out with a new inline xline z spacing but the same origin as the ingoing volumes. This undersampling will be really good to test hyper parameters because one would get a feel for the classification accuracy much quicker 

The second way to improve speed would be to make sure that the to be trained or classified numpy cubes are truly 8 bit integer. Currently there is the option to switch  to 8 bit but it seems not to do the right stuff.

<b>2nd priority </b>is implement 3d augmentation by allowing the user to choose how many and how a sub set of the training cubelets are deformed in 3d (squeezed stretched bent etc)  or adding gaussian noise to them. This would help to make more abstract models


<b>3rd priority </b> is to implement tensor board or other visual tools to see how good the training goes and implement a scikit module to get an accuracy score for withheld training examples 


<b>4th priority </b>would be to implement u nets Gan and other interesting architectures

 

