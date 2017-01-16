import numpy as np
import cv2, json
import matplotlib.pyplot as plt
from mnist import MNIST



class NeuralNet(object):
    def __init__(self, struc=[1,1]):
        '''
        The input struc should be a list (or tupple), in which
            the length of the list is the number of layers (including input and output layers)
            the values of the list correspond to the numbers of neurons of each layer (not including bias neuron)
        '''

        self.struc = struc

        # The structure of weight matrices is static for a given neural network
        # So define the dimension of matrix for each layer
        L = len(struc)
        self.weights = []
        for l in xrange(L-1):

            # The number of rows = the number neurons in the next layer
            rows = struc[l+1]

            # The number of columns = the number of neurons in the present layer + 1 (the 1 is for the bias neuron)
            cols = struc[l] + 1

            # Initialize with random values
            self.weights.append( np.random.rand(rows, cols) )

        self.trained = False

    def __cost_err_grad(self, XT, Y, y, lambda_):
        '''
        Compute the cost and gradients based on current parameters self.weights
            by summing over all input samples
        Applies fully vectorized implementation

        Args:
            XT: The transposed designer matrix, row = feature, column = sample, dtype = np.float

            Y: The outcome matrix, row = category, column = sample, value = (0,1), dtype = np.int

            y: The outcome vector, len(y) = number of samples
               Values of y could be integer 1..K classes
               It is for computational efficiency to have both Y and y as inputs

            lambda_: The regularization parameter

        Returns:
            cost: a scaler;
                  The total cost computed from XT, Y and self.weights

            err: a scaler;
                 Error of predictions

            grad: a list of matrices;
                  The gradients, which is a list of numpy matrices, having identical dimensions as self.weights
        '''

        #------ Prepare neurons and deltas ------#

        # The number of layers of the neural network
        L = len(self.struc)

        # self.neurons --- a list of matrices, in which
        #   each matrix represents a single layer of neuron values, in which
        #     each row --- a neuron
        #     each column --- a sample
        # Therefore the number of rows is static for a given neural network
        #   while the number of columns is dynamic which depends on the input samples
        # Because the structure of each matrix is dynamic, here
        #   I instantiate a list of 'None's without defining the dimensions
        self.neurons = [None] * L

        # Like self.neurons, the structure self.deltas is also dynamic
        # So instantiate a list of 'None's
        self.deltas  = [None] * L

        #------ Forward propagation ------#

        self.neurons[0] = XT
        for l in xrange(L-1):
            # Add bias neurons as the first row
            shape = (1, self.neurons[l].shape[1])
            ones = np.ones(shape, np.float)

            self.neurons[l] = np.concatenate((ones, self.neurons[l]), axis=0)

            # Multiply by the weight matrix to propagate forward
            net_vals = np.dot(self.weights[l], self.neurons[l])

            self.neurons[l+1] = self.__sigmoid(net_vals)

        #------ Compute cost and error ------#

        # Output neurons, row = neuron (category), column = sample
        output = self.neurons[-1]

        # Cost, a scaler
        cost = np.mean(
            - Y * np.log(output) - (1 - Y) * np.log(1 - output)
        )

        # Predictions, a vector with length of sample size
        predictions = np.argmax(output, axis=0)

        # Error of the predictions
        err = 1 - np.mean(np.equal(y, predictions))

        #------ Back propagation ------#

        self.deltas[L-1] = output - Y
        for l in xrange(L-2, 0, -1):
            A = self.neurons[l]
            W_t = np.transpose(self.weights[l])
            C = np.dot(W_t, self.deltas[l+1])
            self.deltas[l] = A * (1 - A) * C
            self.deltas[l] = self.deltas[l][1:, :]

        #------ Compute gradients ------#

        n_samples = XT.shape[1]

        grad = []
        for l in xrange(L-1):
            A_t = np.transpose(self.neurons[l])
            grad.append(
                np.dot(self.deltas[l+1], A_t) / n_samples
            )

        return cost, err, grad

    def __sigmoid(self, z):
        '''
        Args:
            z: a numpy array

        Returns:
           The logistic function, element-wise
        '''

        return 1 / (1 + np.exp(-z))

    def __cost_error_gradient(self, XT, Y, y, lambda_, n_seg):
        '''
        This is a wrapper of the method self.__cost_err_grad().
        Sometimes the sample size could be too large to be computed at once.
        This method divides by the sample (columns) into a list of numpy matrices.
        Iterates over the list, and then average the cost, errors and gradients
            to pool the computation results.

        Args:
            XT: refer to self.__cost_err_grad()
            Y:
            y:
            lambda_:
            n_seg: number of segments for dividing the samples

        Returns:
            cost: a scaler; refer to self.__cost_err_grad()
            err: a scaler
            grad: a list of matrices
        '''

        XTs, Ys, ys = self.__divide_dataset(XT, Y, y, n_seg)

        cost, err, grad = self.__cost_err_grad(XTs[0], Ys[0], ys[0], lambda_)

        if n_seg > 1:
            for s in xrange(1, n_seg):

                cost_, err_, grad_ = self.__cost_err_grad(XTs[s], Ys[s], ys[s], lambda_)
                cost = cost + cost_
                err = err + err_
                for l in xrange(len(grad)):
                    grad[l] = grad[l] + grad_[l]

        cost = cost / n_seg
        err = err / n_seg
        for l in xrange(len(grad)):
            grad[l] = grad[l] / n_seg

        return cost, err, grad

    def __divide_dataset(self, XT, Y, y, nd):

        if nd == 1:
            return [XT], [Y], [y]

        elif nd > 1:
            n_samples = len(y)

            size = n_samples / nd

            XTs = [None] * nd
            Ys = [None] * nd
            ys = [None] * nd

            for s in xrange(nd-1):
                A = size * s
                B = size * (s+1)
                XTs[s] = XT[:, A:B]
                Ys[s] = Y[:, A:B]
                ys[s] = y[A:B]

            A = size * (nd-1)
            XTs[-1] = XT[:, A:]
            Ys[-1] = Y[:, A:]
            ys[-1] = y[A:]

            return XTs, Ys, ys

    def __divide_mat(self, mat, nd, axis):

        if nd == 1:
            return [mat]

        elif nd > 1:
            # If axis == 0 then divide by rows
            if axis == 0:
                n_samples = mat.shape[0]

                size = n_samples / nd

                mats = [None] * nd

                for s in xrange(nd-1):
                    A = size * s
                    B = size * (s+1)
                    mats[s] = mat[A:B, :]

                A = size * (nd-1)
                mats[-1] = mat[A:, :]

            # If axis == 1 then divide by columns
            if axis == 1:
                n_samples = mat.shape[1]

                size = n_samples / nd

                mats = [None] * nd

                for s in xrange(nd-1):
                    A = size * s
                    B = size * (s+1)
                    mats[s] = mat[:, A:B]

                A = size * (nd-1)
                mats[-1] = mat[:, A:]

            return mats

    def gradient_descent(self, X, y, n_iter, alpha, lambda_=0, n_bat=1, n_seg=1):
        '''
        Args:
            X: The designer matrix, row = sample, column = feature dimension, dtype = np.float

            y: The outcome vector, len(y) = number of samples
               Values of y could be integer 1..K classes

            n_iter: number of iterations for gradient descent. In each iteration the whole
                    training samples is computed over.

            alpha: The learning rate

            lambda_: The regularization parameter

            n_bat: The number of batches to compute gradient descent update for each iteration.

            n_seg: The number of segments for each batch of data to be processed.
                   n_seg is mainly for the consideration of memory limitation that
                   sometimes the sample size is too large for a single round of vectorized implementation.
        '''

        #------ Dimension checking ------#

        # The number of input features (columns) of X should match the input layer
        if X.shape[1] != self.struc[0]:
            return

        # The max of y, i.e. the last category, should not be greater than
        #   the number of neurons of the output layer.
        # For example, if the output layer has N neurons,
        #   then max(y) should not be greater than N - 1, as y should belong to 1..(N-1)
        if max(y) > self.struc[-1] - 1:
            return

        # n_samples == nrows of X == length of y
        if X.shape[0] != len(y):
            return

        #------ Data preparation ------#

        XT = np.transpose(X)

        n_output = self.struc[-1] # number of output neurons
        n_samples = len(y)

        # Prepare the Y matrix
        #   Each row: an output neuron = an output category
        #   Eahc column: a sample
        Y = np.zeros((n_output, n_samples), np.int)
        for ith_sample, y_i in enumerate(y):
            Y[y_i, ith_sample] = 1

        # The number of layers of the neural network
        L = len(self.struc)

        if not self.trained:
            # To prevent blow-up of numeric calculation when the number of features is large,
            # divide all weights with n_features
            n_features = X.shape[1]
            for i in xrange(len(self.weights)):
                self.weights[i] = self.weights[i] / n_features

        #------ Gradient descent loop ------#

        costs = []
        errors = []

        for i in xrange(n_iter):

            XTs, Ys, ys = self.__divide_dataset(XT, Y, y, n_bat)

            avg_cost = 0
            avg_err = 0

            for b in xrange(n_bat):

                cost, err, grad = self.__cost_error_gradient(XTs[b], Ys[b], ys[b], lambda_, n_seg)

                for l in xrange(L-1):
                    self.weights[l] = self.weights[l] - alpha * grad[l]

                avg_cost = avg_cost + cost / n_bat
                avg_err = avg_err + err / n_bat

            print 'iter {}, cost={}, error={}'.format(str(i), str(avg_cost), str(avg_err))

            costs.append(avg_cost)
            errors.append(avg_err)

        return costs, errors

    def compute_cost_error(self, X, y, lambda_=0, n_seg=1):
        '''

        X:
        y:
        lambda_:
        n_seg:

        '''

        #------ Dimension checking ------#

        # The number of input features (columns) of X should match the input layer
        if X.shape[1] != self.struc[0]:
            return

        # The max of y, i.e. the last category, should not be greater than
        #   the number of neurons of the output layer.
        # For example, if the output layer has N neurons,
        #   then max(y) should not be greater than N - 1, as y should belong to 1..(N-1)
        if max(y) > self.struc[-1] - 1:
            return

        # n_samples == nrows of X == length of y
        if X.shape[0] != len(y):
            return

        #------ Data preparation ------#

        XT = np.transpose(X)

        n_output = self.struc[-1] # number of output neurons
        n_samples = len(y)

        # Prepare the Y matrix
        #   Each row: an output neuron = an output category
        #   Eahc column: a sample
        Y = np.zeros((n_output, n_samples), np.int)
        for ith_sample, y_i in enumerate(y):
            Y[y_i, ith_sample] = 1

        # The number of layers of the neural network
        L = len(self.struc)

        #------ Compute cost and error ------#

        cost, err, grad = self.__cost_error_gradient(XT, Y, y, lambda_, n_seg)

        return cost, err

    def save(self, filename='neural_net.json'):

        data = {}

        data['readme'] ='''
        'struc':
        The structure of the neural net, which should be a list (or tupple), in which
            the length of the list is the number of layers (including input and output layers)
            the values of the list correspond to the numbers of neurons of each layer (not including bias neuron)

        'weights':
        A list of weight matrices corresponding to the stucture of the neural net.
        Each matrix is in the form of list converted from a 2D numpy matrix.
        '''

        data['struc'] = self.struc

        data['weights'] = [W.tolist() for W in self.weights]

        with open(filename, 'w') as outfile:
            json.dump(data, outfile)

    def load(self, filename):

        with open(filename, 'r') as fh:
            data = fh.read()

        data = json.loads(data)

        self.struc = data['struc']
        self.weights = [np.array(W, np.float) for W in data['weights']]

        self.trained = True

    def predict(self, X, n_seg=1):
        '''
        Args:
            X: The designer matrix, row = sample , column = feature, dtype = np.float

        Returns:
            1D numpy vector of predicted categories
            The length of the vector = the number of samples
        '''

        #------ Prepare neurons ------#

        # The number of layers of the neural network
        L = len(self.struc)

        # Empty neurons, should be a list of numpy matrices after forward propagation
        self.neurons = [None] * L

        #------ Segment data ------#

        XT = np.transpose(X)
        # Divide the input numpy matrix into a list of numpy matrices
        #     to process the data in segmented chunks
        XTs = self.__divide_mat(mat=XT, nd=n_seg, axis=1) # axis=1: segment the matrix by columns

        #------ Forward propagation ------#

        predictions = np.array([], np.int)

        # Process the input data by segmented chunks
        for XT in XTs:
            self.neurons[0] = XT

            # Forward propagation from layer 0 to L-1
            for l in xrange(L-1):
                # Add bias neurons as the first row
                shape = (1, self.neurons[l].shape[1])
                ones = np.ones(shape, np.float)
                self.neurons[l] = np.concatenate((ones, self.neurons[l]), axis=0)

                # Multiply by the weight matrix to propagate forward
                net_vals = np.dot(self.weights[l], self.neurons[l])

                self.neurons[l+1] = self.__sigmoid(net_vals)

            # The max out probability in each column, along all rows (axis=0)
            pred = np.argmax(self.neurons[-1], axis=0)

            # Append the prediction result for the current chunk of data
            predictions = np.concatenate((predictions, pred))

        return predictions

    def predict_single(self, x):
        '''
        Args:
            x: a vector of a single sample, dtype = np.float

        Returns:
            category: int scaler
            probability: float scaler
        '''

        #------ Prepare neurons ------#

        # The number of layers of the neural network
        L = len(self.struc)

        # Empty neurons, should be a list of numpy matrices after forward propagation
        self.neurons = [None] * L

        #------ Forward propagation ------#

        length = x.shape[0]
        self.neurons[0] = np.reshape(x, (length, 1)) # Single column vector

        # Forward propagation from layer 0 to L-1
        for l in xrange(L-1):
            # Add bias neurons as the first row
            shape = (1, self.neurons[l].shape[1])
            ones = np.ones(shape, np.float)
            self.neurons[l] = np.concatenate((ones, self.neurons[l]), axis=0)

            # Multiply by the weight matrix to propagate forward
            net_vals = np.dot(self.weights[l], self.neurons[l])

            self.neurons[l+1] = self.__sigmoid(net_vals)

        # The max probability in the single column, along all rows (axis=0)
        category = np.argmax(self.neurons[-1], axis=0)
        probability = np.max(self.neurons[-1], axis=0)

        return {'category'   : int(category),
                'probability': float(probability)}



class MNISTLoader(MNIST):
    def __init__(self, path):
        super(MNISTLoader, self).__init__(path)

    def load_data(self, set):
        if set == 'train':
            train_set = self.load_training()

            # It is known that the training set of MNIST has 60000 samples,
            #   so the designer matrix X should have 60000 rows
            # Each handwritten image is 28 x 28 = 784 pixels
            #   so X should have 784 columns (features)

            n_samples = 60000
            n_features = 784

            X = np.zeros( (n_samples, n_features), np.float )
            for i in xrange(n_samples):
                X[i, :] = train_set[0][i] # Each row is a sample
            X = X / 255 # Normalize data

            y = np.array(train_set[1])

            return X, y

        elif set == 'test':
            test_set = self.load_testing()

            # It is known that the test set of MNIST has 10000 samples,
            #   so the designer matrix X should have 10000 rows

            n_samples = 10000
            n_features = 784

            X = np.zeros( (n_samples, n_features), np.float )
            for i in xrange(n_samples):
                X[i, :] = test_set[0][i] # Each row is a sample
            X = X / 255 # Normalize data

            y = np.array(test_set[1])

            return X, y



if __name__ == '__main__':

    loader = MNISTLoader('mnist_data')

    X, y = loader.load_data('train')

    # Show some digits from the training data
    for i in xrange(100):
        img = np.reshape(X[i, :], (28, 28))
        cv2.imshow('digit', img)
        cv2.waitKey(500)

