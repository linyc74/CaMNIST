"""
Version 0.3

Give the option to compute cost or not for each epoch
"""

__version__ = '0.3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math, cv2, json, time

class NeuralNet(object):
    """
    There are two main groups of attributes for this neural network model.

    --- Group 1 --- Model attributes ---
    Attributes defining the architecture of the network.
    With these attributes, forward propagation and prediction can work.
        self.X : input units       --- tf.placeholder
        self.Y : labels            --- tf.placeholder
        self.W : weight matrices   --- dictionary, {l: tf.Variable} where l is an int of the layer number
        self.b : bias vectors      --- dictionary, {l: tf.Variable} where l is an int of the layer number
        self.Z : net values        --- dictionary, {l: tf.tensor}   where l is an int of the layer number
        self.A : activation values --- dictionary, {l: tf.tensor}   where l is an int of the layer number

    --- Group 2 --- Optimization attributes ---
    Attributes for tensorflow to optimize the parameters self.W and self.b
        self.cost      : tf.tensor, a scalar
        self.error     : tf.tensor, a scalar
        self.optimizer : tf.train.optimizer().minimize()
    """
    def __init__(self, device='gpu'):
        """
        :parameter
            device: str, ('cpu', 'gpu')
        """
        # Two state attributes
        self.isModelSet = False # If true => Can predict
        self.isOptimizationSet = False # If true => Can train and predict

        if device == 'cpu':
            self.device = '/cpu:0'
        elif device == 'gpu':
            self.device = '/gpu:0'

    def __initialize_parameters(self, layer_dims, seed=1):
        """
        :parameter
            layer_dims:
                list of int, number of units in each layer
            seed:
                scalar int, seed for numpy.random

        :returns
            weights:
                dictionary, {l: tf.Variable} where l is an int of the layer number
            biases:
                dictionary, {l: tf.Variable} where l is an int of the layer number
        """
        np.random.seed(seed=seed)

        # For example, if there are 3 elements in self.layer_dims, then it's a 2-layered network
        # So L = 3 - 1 = 2
        L = len(layer_dims) - 1

        weights = {}
        biases = {}
        for l in range(1, L+1):
            # n_rows = number of neurons in the lth layer
            # n_cols = number of neurons in the (l-1)th layer
            rows, cols = layer_dims[l], layer_dims[l-1]

            W = np.random.rand(rows, cols) * 2 - 1 # uniform random float number betwee -1 and 1
            W = W / math.sqrt(cols) # Xavier initialization
            b = np.zeros((rows, 1), np.float32)

            weights[l] = tf.Variable(W, name='W'+str(l), dtype=tf.float32)
            biases[l]  = tf.Variable(b, name='b'+str(l), dtype=tf.float32)

        return weights, biases

    def __load_parameters(self, W, b):
        """
        :parameter
            W:
                list of 2D matrices, each matrix is a list of list of float
            b:
                list of column vectors, each column vector is a list of list of float

        :returns
            layer_dims:
                list of int, number of units in each layer
            weights:
                dictionary, {l: tf.Variable} where l is an int of the layer number
            biases:
                dictionary, {l: tf.Variable} where l is an int of the layer number
        """
        layer_dims = [np.array(W[0]).shape[1]]
        weights = {}
        biases = {}
        for i in range(len(W)):
            l = i + 1
            weights[l] = tf.Variable(np.array(W[i], np.float32), name='W'+str(l), dtype=tf.float32)
            biases[l]  = tf.Variable(np.array(b[i], np.float32), name='b'+str(l), dtype=tf.float32)
            layer_dims.append(
                np.array(b[i]).shape[0] # b[l] is n_l x 1
            )
        return layer_dims, weights, biases

    def __create_placeholders(self, layer_dims):
        """
        :parameter
            layer_dims:
                list of int, number of units in each layer

        :returns
            X:
                tf.placeholder for input units
            Y:
                tf.placeholder for labels
        """
        n_features = layer_dims[0]
        n_classes = layer_dims[-1]
        X = tf.placeholder(shape=(n_features, None), dtype=tf.float32, name='X')
        Y = tf.placeholder(shape=(n_classes, None) , dtype=tf.float32, name='Y')
        return X, Y

    def __foward_propagation(self, X, W, b, activation):
        """
        :parameter
            X:
                tf.placeholder for input units
            W:
                dictionary, {l: tf.Variable} where l is an int of the layer number
            b:
                dictionary, {l: tf.Variable} where l is an int of the layer number
            activation:
                list of str, ('relu', 'tanh', ' sigmoid', 'softmax'), len(activation) = L

        :returns
            Z:
                net values, dictionary, {l: tf.tensor} where l is an int of the layer number
            A:
                activation values, dictionary, {l: tf.tensor} where l is an int of the layer number
        """
        L = len(W.keys())
        A = {}
        Z = {}
        A[0] = X
        for l in range(1, L+1):
            # From A[l-1] to Z[l]
            Z[l] = tf.matmul(W[l], A[l-1]) + b[l]

            # Activation from Z[l] to A[l]
            name = activation[l-1] # activation is 0-based index, so it's l-1
            activate = getattr(tf.nn, name)
            if name in ['relu', 'tanh', 'sigmoid']:
                A[l] = activate(Z[l])
            elif name == 'softmax':
                A[l] = activate(Z[l], dim=0) # softmax is different that it requires dim
        return Z, A

    def __cost_error(self, lambd, W, A, Y, activation):
        """
        :parameter
            lambd:
                L2 regularization lambda
            W:
                dictionary, {l: tf.Variable} where l is an int of the layer number
            A:
                activation values, dictionary, {l: tf.tensor} where l is an int of the layer number
            Y:
                tf.placeholder for labels, one-hot encoding
            activation:
                list of str, ('relu', 'tanh', ' sigmoid', 'softmax'), len(activation) = L

        :returns
            cost:
                tf.tensor, a scalar
            error:
                tf.tensor, a scalar
        """
        ### Cost with Frobenius (L2) norm
        # Here I implement the cross-entropy of softmax and logistic with lower-level functions
        #   because I found tf.nn.softmax_cross_entropy_with_logits() could porduce
        #   error when the input sample batch is either very large (~60000) or small (~16)
        L = len(W.keys())
        if activation[L-1] == 'softmax':
            cost = - Y * tf.log(A[L])
        elif activation[L-1] == 'sigmoid':
            cost = - Y * tf.log(A[L]) - (1 - Y) * tf.log(1 - A[L])

        cost = tf.reduce_mean(cost) # cost becomes a scalar
        lambd = tf.constant(lambd)
        for l in range(1, L+1): # Layer by layer, add L2 norm of weights
            cost = cost + lambd * tf.nn.l2_loss(W[l])

        ### Error
        y_prep = tf.argmax(A[L], axis=0)
        y = tf.argmax(Y, axis=0) # convert one-hot encoding matrix back to a 1-D label vector
        error = 1 - tf.reduce_mean(tf.cast(tf.equal(y, y_prep), tf.float32))

        return cost, error

    def __set_optimizer(self, optimizer_name, learn_rate, cost):
        """
        :parameter
            optimizer_name:
                str, the optimizer name in tf.train
            learn_rate:
                scalar float, the learning rate
            cost:
                tf.tensor, a scalar

        :returns
            tf.train.optimizer(learn_rate).minimize(cost)
        """
        optimizer = getattr(tf.train, optimizer_name)
        return optimizer(learn_rate).minimize(cost)

    def __check_dims(self, X, y=None):
        """
        :parameter
            X:
                The input 2-D matrix, float32, row = feature, column = sample, dtype = np.float
            y:
                The label 1-D vector, float32, length = number of samples
                Values of y could be integer 1..K classes
        
        :returns
            boolean, indicating input data dimension is correct or not
        """

        # The number of input features (rows) of X == the input layer dimension
        if not X.shape[0] == self.layer_dims[0]:
            print('The number of input features (rows) of X does not equal to the input layer dimension')
            return False

        if y is None:
            return True

        # The max of y, i.e. the last category, should not be greater than
        #   the number of neurons of the output layer.
        # For example, if the output layer has N neurons,
        #   then max(y) should not be greater than N - 1, as y should belong to 1..(N-1)
        if max(y) > self.layer_dims[-1] - 1:
            print('The max value of the label vector y is greater than the number of output neurons')
            return False

        # n_samples == n_cols of X == length of y
        if not X.shape[1] == len(y):
            print('The number of input samples (columns) of X does not equal to the number of samples (length) of y')
            return False

        return True

    def __one_hot(self, y, depth):
        """
        :parameter
            y:
                The label 1-D vector, np.uint8, length = number of samples
                Values of y could be integer 1..K classes
            depth:
                number of categories, i.e. number of rows of the output matrix

        :returns
            one-hot matrix, dtype=np.int, each row = category, each column = sample
        """
        Y = np.zeros((depth, len(y)), np.int)
        for i in range(len(y)):
            Y[y[i], i] = 1
        return Y

    def __init_sess(self):
        """
        A method initializing self.sess
        """
        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        self.sess.run(init)

    def set_model(self, layer_dims, activation):
        """
        Method setting the architecture of the neural network.

        :parameter
            layer_dims:
                list of int, number of units of each layer, len(layer_dims) = L+1
            activation:
                list of str, ('relu', 'tanh', ' sigmoid', 'softmax'), len(activation) = L
        """
        if self.isModelSet:
            print('Model already set and cannot be changed.')
            return

        for key, val in locals().items():
            if key != 'self':
                setattr(self, key, val)

        with tf.device(self.device):
            self.W, self.b = self.__initialize_parameters(layer_dims)
            self.X, self.Y = self.__create_placeholders(layer_dims)
            self.Z, self.A = self.__foward_propagation(self.X, self.W, self.b, activation)

        self.__init_sess()
        self.isModelSet = True

    def load_model(self, filename):
        """
        Method loading saved model.
        """
        with open(filename, 'r') as fh:
            model = json.loads(fh.read())

        weights        = model['weights']
        biases         = model['biases']
        activation     = model['activation']
        learn_rate     = model['learn_rate']
        lambd          = model['lambd']
        keep_prob      = model['keep_prob']
        optimizer_name = model['optimizer_name']

        self.activation = activation
        with tf.device(self.device):
            self.layer_dims, self.W, self.b = self.__load_parameters(W=weights, b=biases)
            self.X, self.Y = self.__create_placeholders(self.layer_dims)
            self.Z, self.A = self.__foward_propagation(self.X, self.W, self.b, activation)

        self.isModelSet = True

        # Re-set the computation graph
        self.set_optimization_parameters(learn_rate=learn_rate, lambd=lambd,
                                         keep_prob=keep_prob, optimizer_name=optimizer_name)

    def save_model(self, filename='nn.json'):
        """
        Method saving the trained model.

        Model parameters:
            weights, biases and activation
        Optimization parameters:
            learn_rate, lambd, keep_prob, optimizer_name
        """

        L = len(self.W.keys())
        weights = []
        biases = []
        # Convert dictionary of tf.Variables (2-D matrix)
        #   to list of 2-D matrices (each matrix is a list of list)
        for l in range(1, L+1):
            weights.append(
                self.sess.run(self.W[l]).tolist()
            )
            biases.append(
                self.sess.run(self.b[l]).tolist()
            )

        model = {'weights'       : weights,
                 'biases'        : biases,
                 'activation'    : self.activation,
                 'learn_rate'    : self.learn_rate,
                 'lambd'         : self.lambd,
                 'keep_prob'     : self.keep_prob,
                 'optimizer_name': self.optimizer_name}

        with open(filename, 'w') as fh:
            json.dump(model, fh)

    def set_optimization_parameters(self, learn_rate=0.001, lambd=0., keep_prob=1., optimizer_name='AdamOptimizer'):
        """
        :parameter
            learn_rate     : scalar float, the learning rate
            optimizer_name : str, the optimizer name in tf.train
            lambd          : scalar float, the regularization parameter
            keep_prob      : scalar float, keep probability for dropout
        """
        if not self.isModelSet:
            print('Neural network model not defined.')
            return

        for key, val in locals().items():
            if key != 'self':
                setattr(self, key, val)

        with tf.device(self.device):
            self.cost, self.error = self.__cost_error(lambd, self.W, self.A, self.Y, self.activation)
            self.optimizer = self.__set_optimizer(optimizer_name, learn_rate, self.cost)

        self.__init_sess()
        self.isOptimizationSet = True

    def train(self, X, y, num_epochs=1000, batch_size=None, compute_cost=True, stop_thres=0.):
        """
        Public method training the model parameters self.W and self.b.

        :parameter
            X:
                The input 2-D matrix, float32, row = feature, column = sample, dtype = np.float
            y:
                The label 1-D vector, uint8, length = number of samples
                Values of y could be integer 1..K classes
            num_epochs:
                scalar int, number of epoch for gradient descent
            batch_size:
                scalar int, the size (# samples) of each mini batch
                If None, then the batch_size will be set to the whole training set
            compute_cost:
                boolean, compute cost or not for each epoch
                Defualt True. If set to False then the stop_thres is not considered.
            stop_thres:
                scalar float, the stopping threshold for training iteration

        :returns
            None
        """
        # Sanity check that the model and optimization are both set.
        # That is, the computation graph is complete.
        if not all([self.isModelSet, self.isOptimizationSet]):
            return

        # Check dimensions
        if not self.__check_dims(X, y):
            return

        # Training parameters
        n_samples = len(y)
        if batch_size == None or batch_size > n_samples:
            batch_size = n_samples

        # Prepare data
        X_train = X
        Y_train = self.__one_hot(y, depth=self.layer_dims[-1])

        # Shuffle data (columns)
        np.random.seed(seed=1)
        rn = np.random.permutation(np.arange(n_samples))
        X_train, Y_train = X_train[:, rn], Y_train[:, rn]

        # Prepare placeholders
        input_units, labels = self.X,  self.Y

        feed_dict = {input_units:X_train, labels:Y_train}
        epoch_cost_prev = self.sess.run(self.cost, feed_dict=feed_dict)
        t0 = time.time()
        for epoch in range(num_epochs):

            n_batches = int(n_samples/batch_size)

            # For each mini-batch, run a session of training
            for bat in range(n_batches):
                a = (bat  ) * batch_size
                b = (bat+1) * batch_size
                X_bat, Y_bat = X_train[:, a:b], Y_train[:, a:b]
                feed_dict = {input_units:X_bat, labels:Y_bat}
                self.sess.run(self.optimizer, feed_dict=feed_dict)

            # Train with the last few samples not in those mini-batches
            a = n_batches * batch_size
            X_bat, Y_bat = X_train[:, a:], Y_train[:, a:]
            feed_dict = {input_units:X_bat, labels:Y_bat}
            self.sess.run(self.optimizer, feed_dict=feed_dict)

            if not compute_cost:
                print(
                    '{}\ttime={} s'.format(epoch+1, time.time()-t0)
                )
                continue

            # For each epoch, compute cost and error for the whole training set
            feed_dict = {input_units:X_train, labels:Y_train}
            epoch_cost, epoch_err = self.sess.run([self.cost, self.error], feed_dict=feed_dict)
            print(
                '{}\tcost={}\terror={}%\ttime={} s'.format(
                epoch+1, epoch_cost, epoch_err*100, time.time()-t0)
            )

            if (epoch_cost_prev - epoch_cost) / epoch_cost_prev < stop_thres:
                break
            epoch_cost_prev = epoch_cost

    def compute_cost_error(self, X, y):
        """
        :parameter
            X:
                The input 2-D matrix, float32, row = feature, column = sample, dtype = np.float
            y:
                The label 1-D vector, uint8, length = number of samples
                Values of y could be integer 1..K classes

        :returns
            cost:
                scalar float, cross-entropy L2-norm cost
            error:
                scalar float, error rate
        """
        # Sanity check that the model and optimization are both set.
        # That is, the computation graph is complete.
        if not all([self.isModelSet, self.isOptimizationSet]):
            return

        # Check dimensions
        if not self.__check_dims(X, y):
            return

        # Prepare data
        Y = self.__one_hot(y, depth=self.layer_dims[-1])

        # Prepare placeholders
        input_units, labels = self.X,  self.Y

        feed_dict = {input_units:X, labels:Y}
        cost, err = self.sess.run([self.cost, self.error], feed_dict=feed_dict)

        return cost, err

    def predict(self, X):
        """
        :parameter
            X:
                The input 2-D matrix, float32, row = feature, column = sample, dtype = np.float

        :returns a dictionary {'category': np.array, 'probability': np.array}
            category:
                1-D numpy vector of int
            probability:
                1-D numpy vector of float, softmax or sigmoid probability
        """
        # Sanity check that the model computation graph is set.
        # The optimization graph (cost) is not required.
        if not self.isModelSet:
            return

        # Check dimensions
        if not self.__check_dims(X):
            return

        L = len(self.layer_dims) - 1
        input_units = self.X
        softmax = self.sess.run(self.A[L], feed_dict={input_units:X})
        pred = {'category'   : softmax.argmax(axis=0),
                'probability': softmax.max(axis=0)}
        return pred


