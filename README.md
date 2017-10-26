# MalenoV_nD
<h2> Tool for training &amp;  classifying 3D (4D, nD) SEGY seismic facies using deep neural networks</h2>

•	MalenoV reads standard 3D SEGY seismic and performs a 3D neural network architecture of choice on a given set of classification data points (facies annotation /supervision).  It then uses the learned weights and filters of the neural network to classify seismic at any other location in the seismic cube into the facies classes that have been previously been defined by the user. Finally the facies classification is written out as a SEGY cube with the same dimensions as the input cube.

The tool reads can handle n seismic input cubes (offest stacks, 4D data, attributes, etc) and n number of facies training datasets. The more input seismic cubes are trained / classified  simultaneously the more memory is needed and the slower training /classification will be (linear scaling).

•	MalenoV was created as part of the Summer project of Charles Rutherford Ildstad in ConocoPhillips in July 2017.

•	The tool was inspired by the work of Anders U. Waldeland who showed at that he was successfully classifying the seismic facies of salt using 3D convolutional networks (http://earthdoc.eage.org/publication/publicationdetails/?publication=88635). 

•	Currently a 5 layer basic 3D convolutional network is implemented but this can be changed by the users at liberty. 

•	This tool is essentially an I/O function for machine learning with seismic. Better neural architectures for AVO, rock property prediction, fault classifications are to be implemented. (Think Unet resnet or GAN).

•	The tool is public with a GNU Lesser General Public License v3.0

•	The tool  has been updated to handle multiple input volumes (offest stacks, 4D seismic) for better classification results and more fun

<h3>The User Manual and seismic data for the tool can be found here</h3>
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

