
	+----------------------------------------------------------------------------------------+
	|	HOW TO RUN																			                                       |
	+----------------------------------------------------------------------------------------+
	|																						                                             |
	|	Before running my application, you must install the following:                   			 |
	|																						                                             |
	|	pip install torch torchvision torchaudio								                        			 |
	|	pip install matplotlib scikit-learn pandas flask flask-cors pillow				          	 |
	|																                                            						 |
	|	Once you have these installed, you can navigate to the 						                 		 |
	|	../BirdSpeciesIdentifier/Project directory run the following command:				           |
	|																						                                             |
	|	python app.py																		                                       |
	|																						                                             |
	|	Then click the URL it prints and upload your own bird photos.						               |
	|																						                                             |
	|	The application will then display three possible bird species guesses back to you.     |
	|																						                                             |
	|	Some sample photos that are within the 200 species have been provided under the 	     |
	|	directory ../BirdSpeciesIdentifier/SamplePhotos										                     |
	+----------------------------------------------------------------------------------------+

	+----------------------------------------------------------+
	|	DATA SET USED										                         |
	+----------------------------------------------------------+
	|	Caltech-UCSD Birds-200-2011 (CUB-200-2011)			         |
	|	https://www.vision.caltech.edu/datasets/cub_200_2011/    |
	+----------------------------------------------------------+

	+----------------------------------------------------------------------------------------+
	|	PYTHON FILE DETAILS																	                                   |
	+----------------------------------------------------------------------------------------+
	|	app.py:																				                                         |
	|		This file opens the Flask application and supports users uploading photos, and       |
	|		then loads in the model to predict 3 bird species and display the result to the      |
	|		webpage.																		                                         |
	|																						                                             |
	|	utils.py:																			                                         |
	|		This file loads the bird classification model, applies image transforms, and  	     |
	|		predicts the top 3 species with example images. This file is used to support 	       |
	|		app.py.																			                                         |
	|																						                                             |
	|	Evaluate.py:																		                                       |
	|		This file is used to run accuracy tests for the top-1, top-3, and top-5 		         |
	|		predictions based on the test set from the data. To run this, you must download      |
	|		the data set and add it into the ../BirdSpeciesIdentifier/Project directory.	       |
	|																						                                             |
	|	Train.py:																			                                         |
	|		This file is how I trained the bird_model.pth model I provided for the program 	     |
	|		to run. To run this, you must download the data set and add it into the 		         |
	|		../BirdSpeciesIdentifier/Project directory.										                       |
	+----------------------------------------------------------------------------------------+
