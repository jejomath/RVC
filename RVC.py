###########################################################################
##
## Random Voronoi Cells
##
## Copyright 2013 Jesse Johnson <jejo.math@gmail.com>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You can find a copy of the GNU General Public License
## at <http://www.gnu.org/licenses/>.
##
###########################################################################

import scipy, random
import sklearn.metrics as metrics
from operator import itemgetter

class randomVoronoiCells:
    
    def __init__(self, reset_threshold=0, centroid_sharing=1, verbose = 0):
        # Save the setting passed in by __init__
        self.reset_threshold = reset_threshold
        self.centroid_sharing = centroid_sharing
        self.verbose = verbose
        
        # Create the empty list of saved centroids
        self.saved_centroids = []
        self.saved_centroid_ratios = []
        
        
    def fit(self, data, labels, centroid_weights = [], iteration_number = 3, 
            seed_labels = [], label_index = [], point_weights = []):
            
        # Save the data and labels
        if data.shape[0] != labels.shape[0]:
            print "Data size and labels size must match."
            return
        else:
            self.data = data
            self.labels = labels
        
        # Check that centroid_weights have been specified
        if centroid_weights == []:
            print "Cannot fit data without specified list of centroid_weights."
            return
        else:
            self.centroid_weights = centroid_weights
            
        # Generate the label index if it hasn't been provided
        if label_index == []:
            self.label_index = list(set(labels))
        else:
            self.label_index = list(label_index)
            
        # Generate the seed labels if they haven't been specified
        if seed_labels == []:
            self.seed_labels = [centroid_weights[i].index(max(centroid_weights[i])) \
                                for i in range(len(centroid_weights))]
        else:
            self.seed_labels = seed_labels
                    
        # Set point weights to all ones if they are not specified
        if point_weights == []:
            self.point_weights = scipy.ones(data.shape[0])
        else:
            self.point_weights = point_weights

        # Create lists of the data points with each label for recruiting seeds
        self.seed_recruit_list = []
        for l in self.label_index:
            self.seed_recruit_list.append([i for i in range(labels.shape[0]) \
                                            if labels[i] == l])
            
        # Create initial list of random centroids
        self.centroids = [list(self.data[self.random_seed(self.seed_labels[i]),:]) \
                            for i in range(len(self.centroid_weights))]
        self.centroids = scipy.array(self.centroids)        
                    
        self.assign_points()
        
        for i in range(iteration_number):
            self.move_centroids()
            self.assign_points()
            if self.verbose == 1:
                print i, self.calculate_current_AUC()
            
        self.record_centroids()
                                

    def random_seed(self, index):
        return self.seed_recruit_list[index][random.randrange(len(self.seed_recruit_list[index]))]

    def assign_points(self):
        # The matrix used to calculate the new centroids
        self.assignments = scipy.zeros([self.centroids.shape[0], self.data.shape[0]])
        # The (weighted) number of points assigned to each centroid
        self.centroid_populations = scipy.zeros(self.centroids.shape[0])
        # The proportion of labels in each centroid
        self.ratios = scipy.zeros([self.centroids.shape[0], len(self.label_index)])
        # The index of the centroid assign to each point
        self.point_assignments = scipy.zeros(self.data.shape[0])
        
        # Pre-calculate centroid norms
        cnorms = scipy.zeros(self.centroids.shape[0])
        for i in range(self.centroids.shape[0]):
            cnorms[i] = self.centroids[i,:].dot(self.centroids[i,:].transpose())

        self.cnorms = cnorms
        # Assign points to the centroids
        for i in range(self.data.shape[0]):
            # Calculate the list of distances from point i to each centroid
            dlist = list(cnorms - 2*self.centroids.dot(self.data[i]))
            dlist = [[dlist[k],k] for k in range(len(dlist))]
            dlist.sort(key=itemgetter(0)) # Sorted list of pairs [distance, index]
            
            # Record the first closest centroid for each point and the population
            # ratios of the centroids for only the first closest.
            j = dlist[0][1]
            self.point_assignments[i] = j
            self.ratios[j, self.label_index.index(self.labels[i])] += 1
            # For the move step, record the n-closest centroids, where n
            # is self.centroid_sharing
            for j in [dlist[k][1] for k in range(self.centroid_sharing)]:
                self.assignments[j,i] = self.centroid_weights[j][self.label_index.index(self.labels[i])]
                self.centroid_populations[j] += self.centroid_weights[j][self.label_index.index(self.labels[i])]

        # Convert self.ratios from a count to a set of percentages
        for i in range(self.ratios.shape[0]):
            if sum(self.ratios[i]) > 0:
                self.ratios[i] = self.ratios[i]/sum(self.ratios[i])

    def move_centroids(self):
        self.centroids = self.assignments.dot(self.data)
        normalizer = [1/self.centroid_populations[i] for i in range(self.centroids.shape[0])]
        self.centroids = scipy.diag(normalizer).dot(self.centroids)

        for i in range(self.centroids.shape[0]):
            if self.centroid_populations[i] <= self.reset_threshold:
                self.centroids[i,:] = self.data[self.random_seed(self.seed_labels[i]),:]

    def record_centroids(self):
        self.saved_centroids.append(self.centroids)
        self.saved_centroid_ratios.append(self.ratios)

    def calculate_current_AUC(self):
        pred = [self.ratios[self.point_assignments[i]][1] for i in range(len(self.labels))]
        actual = [float(self.label_index.index(self.labels[i])) for i in range(len(self.labels))]

        fpr, tpr, thresholds = metrics.roc_curve(actual, pred, pos_label=1)
        auc = metrics.auc(fpr,tpr)
        return auc

    def predict(self, data):
        cnorms = []
        for c in self.saved_centroids:
            cn = scipy.zeros(c.shape[0])
            for j in range(c.shape[0]):
                cn[j] = c[j,:].dot(c[j,:].transpose())
            cnorms.append(cn)

        pred = []
        for i in range(data.shape[0]):
            c = 0
            for j in range(len(self.saved_centroids)):
                dlist = self.saved_centroids[j].dot(data[i])
                dlist = cnorms[j] - 2*dlist
                dlist = list(dlist)
                k = dlist.index(min(dlist))
                c += self.saved_centroid_ratios[j][k][1]
            pred.append(c / len(self.saved_centroids))
            
        return pred
            
            


        
