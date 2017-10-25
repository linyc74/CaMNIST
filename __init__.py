"""
Version 2.0 is a major shift from version 1.0

Changes:
    Implemented the neural network with tensorflow

    Use two neural networks,
        one for telling a digit or not,
        the other for recognizing digit number.
    The one telling digit or not is the model 'exp_17_1024_1_model-lambda1e-09.json'
    The one recognizing digit number is the model 'exp_17_1022_softmax.json'

    The two neural networks were placed in a separate PredictionThread() class.
    PredictionThread() object calls methods in the VideoThread() object to detect digits.

Notes:
    The program runs smoothly and it draws the digits on the image robustly without any error.
    However, what's really hard is the digit recognition.
    In particular the neural network that I trained is not good enough to work in
    a real world scenario. This could be due to the limitation and bias of my training
    samples: MNIST and CIFAR-10. MNIST is too clean and may not accurately represent what
    a digit image looks like.
"""

__version__ = '2.0'

from model import *

def launch():
    app = QtGui.QApplication(sys.argv)
    core = Core()
    sys.exit(app.exec_())

if __name__ == '__main__':
    launch()
