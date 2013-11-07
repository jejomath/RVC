# Random Voronoi Cells

This is a python package implementing the Random Voronoi
cells algorithm. RVC is a classification algorithm using
an ensemble method similar to the popular Random Forests
algorithm. However, rather than combining a number of 
decisions trees, each model in the RVC ensemble is a set of
"centroids" that define a partition of the data space into
Voronoi cells. The algorithm calculates the number of
training points from each label in each Voronoi cell, and
uses these to predict the class of each new data point.

Each set of "centroids" is constructed using a modification
of the K-means algorithm that takes into account data labels.
Rather than take the actual set of centroids of a given 
subset of the data points, each new "centroid" is a 
weighted average, where the weights are determined by the 
labels. Each "centroid" can be given a different set of
rules for assigning weights, as specified by the user.

A preprint with a more detailed description of the 
algorithm is currently being written.

An example script, "test_RVC.py" is included, which runs RVC
on the OQ data set (also included) which was downloaded from
the UCI Machine Learning Repository 
(http://archive.ics.uci.edu/ml/datasets.html)

This implementation of RVC is based on the sci-kit learn model. 
The script uses sci-kit learn's metrics package to calculate
the AUC of each set of Voronoi cells, when the verbose option
is set to 1, so you must have scikit learn. (If you don't want
to install scikit, you can just remove the import statement
at the top of RVC.py and keep verbose set to 0.)

A separate call must be made to rvc.fit for each individual
model in the ensemble. A basic call looks like this:


	rvc = RVC.randomVoronoiCells()
	rvc.fit(data, labels, centroid_weights = [[1,0]]*10+[[0,1]]*10)


The first two variables are an array of data points in vector
format and a list or array of labels. RVC assigns an index
to each class that appears in the labels list, using the
python "set" function. The user can override this by 
passing a list of classes to the fit command with the 
"label_index" parameter.

If you set the verbose option to 1 when creating the
randomVoronoiCells object, it will report the AUC (i.e. the area
under the reciever-operator curve (ROC)) for the training set after 
each step in the K-means process. The included test script
calculates the AUC for both a training set and a separate
(randomly selected) evaluation set. 

Because the modified K-means step does not use true centroids, 
it is not guaranteed to converge, so the algorithm uses a
hard cutoff on the number of steps, rather than a convergence
check. The default number of steps is 3, but this can be
changed with the "iteration" parameter in the "fit" function.


