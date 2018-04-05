import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the 
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            for j in xrange(num_train):
                ###############################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                ###############################################################
                      # solution 1:
                #dists[i, j] = np.sqrt(np.sum(np.square(X[i, :] - self.X_train[j, :])))

                # solution 2:
                dists[i, j] = np.linalg.norm(self.X_train[j, :] - X[i, :])

                ###############################################################
                #                       END OF YOUR CODE                            #
                ###############################################################
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            ###################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            ###################################################################
            # solution 1
            #dists[i, :] = np.sqrt(np.sum(np.square(X[i, :] - self.X_train), axis=1))

            # solution 2
            dists[i, :] = np.linalg.norm(self.X_train - X[i, :], axis=1)

            ###################################################################
            #                         END OF YOUR CODE                            #
            ###################################################################
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #######################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #######################################################################

        M = np.dot(X, self.X_train.T)
        te = np.square(X).sum(axis=1)
        tr = np.square(self.X_train).sum(axis=1)
        dists = np.sqrt(-2 * M + np.matrix(tr) + np.matrix(te).T)
        # print("X.type", type(X));
        # print("X.shape = ", X.shape);
        # print("self.X_train.shape = ", self.X_train.shape)
        # print("M.shape = ", M.shape);
        # print("te.shape = ", te.shape);
        # print("tr.shape = ", tr.shape);
        # print("dists.shape = ", dists.shape);
        """
        ('X.type', <type 'numpy.ndarray'>)
        ('X.shape = ', (500L, 3072L))
        ('self.X_train.shape = ', (5000L, 3072L))
        ('M.shape = ', (500L, 5000L))
        ('te.shape = ', (500L,))
        ('tr.shape = ', (5000L,))
        ('dists.shape = ', (500L, 5000L))
		    """
        #######################################################################
        #                         END OF YOUR CODE                              #
        #######################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            ###################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            ###################################################################

            """
            temp = np.argsort(dists[i, :]);
            closest_y = self.y_train[temp[:k]];  #get k nearest label
            #print(closest_y);
            #print(type(closest_y));  #eg:  [5 3 5 2 2]
            """
         
            # solution 1:
            labels = self.y_train[np.argsort(dists[i, :])].flatten()
            # print labels.shape
            closest_y = labels[0:k]
            # print 'k is %d' % k
            """
            
            # solution 2:
            y_indicies = np.argsort(dists[i, :], axis=0)
            closest_y = self.y_train[y_indicies[:k]]
            """
            #print(closest_y.shape)  #(1, 5000)
            #test_closest_y = np.reshape(closest_y, 5000)
            #print(test_closest_y.shape)

            ###################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            ###################################################################
            
            """
            count = np.bincount(closest_y);   #ValueError: object too deep for desired array
            max_index = np.argmax(count);
            y_pred[i] = max_index;
            """

            #c = Counter(closest_y)
            #y_pred[i] = c.most_common(1)[0][0]

            y_pred[i] = np.argmax(np.bincount(closest_y))
            
            ###################################################################
            #                           END OF YOUR CODE                            #
            ###################################################################

        return y_pred