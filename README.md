# MalenoV
Complete tool for training &amp;  classifying 3D SEGY seismic using deep neural networks

•	MalenoV reads standard 3D SEGY seismic and performs a 3D neural network architecture of choice on a given set of classification data points (facies annotation /supervision).  It then uses the learned weights and filters of the neural network to classify seismic at any other location in the seismic cube into the facies classes that have been previously been defined by the user. Finally the facies classification is written out as a SEGY cube with the same dimensions as the input cube

•	MalenoV was created as part of the Summer project of Charles Rutherford Ildstad in ConocoPhillips in July 2017.

•	The tool was inspired by the work of Anders U. Waldeland who showed at that he was successfully classifying the seismic facies of salt using 3D convolutional networks (http://earthdoc.eage.org/publication/publicationdetails/?publication=88635). 

•	Currently a 5 layer basic 3D convolutional network is implemented but this can be changed by the users at liberty

•	The tool is public with a GNU Lesser General Public License v3.0

The User Manual for the tool can be found here
https://goo.gl/wb145Z


Seismic training data for testing the tool:

•	Since there is a limited number of free seismic dataset publicly available we decided to make available the Poseidon (3500km2) seismic dataset acquired for ConocoPhillips Australia including Near, Mid, Far stacks, well data and velocity data

•	The seismic data is available here: https://goo.gl/wb145Z
•	               BEAWRE one 32 bit SEGY File is 100 GB of data

•	There is also inline ,xline, z training data for fault identification 
•	
•	The Dutch government F3 seismic dataset can also be downloaded from the same location including
•	               This data is only 1 GB

•	Training data for multi facies prediction, faults and steep dips
•	Trained neural network models for steep dips and multi facies
