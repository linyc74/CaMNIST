import time, sys, threading
from PyQt4 import QtCore, QtGui
from functools import partial



class Mediator(QtCore.QThread):
    '''
    A purely administrative-logic object.

    Mediator defines the interface to emit signals to the gui object.
    Each signal is defined by a unique str 'signal_name'.
    '''
    def __init__(self, gui_obj):
        super(Mediator, self).__init__()
        self.gui = gui_obj

    def __del__(self):
        self.exiting = True
        self.wait()

    def connect_signals(self, signal_names):
        '''
        Pass signal names to the connect_signals() method in the gui object.
        The gui object does the actual signal-slot connection.
        The gui object also decides what to do when it receives the signal.
        '''
        if isinstance(signal_names, str):
            self.gui.connect_signals( thread=self, signal_name=signal_names )

        elif isinstance(signal_names, list):
            for signal in signal_names:
                self.gui.connect_signals( thread=self, signal_name=signal )

    def emit_signal(self, signal_name, arg=None):
        # The suffix '(PyQt_PyObject)' means the argument to be transferred
        # could be any type of python objects,
        # not limited to Qt objects.
        self.emit(QtCore.SIGNAL( signal_name + '(PyQt_PyObject)' ), arg)



class Controller(object):
    '''
    A purely administrative-logic object, which
    parametrizes and calls the method name in the core object.
    '''
    def __init__(self, core_obj):
        super(Controller, self).__init__()
        self.core = core_obj

    def call_method(self, method_name, arg=None):

        # Find out if the method_name is present in the core object.
        # If not, return.
        try:
            method = getattr(self.core, method_name)

        except Exception as exception_inst:
            print("The controller tries to call method '{}' in 'Core' object,".format(method_name))
            print("but " + str(exception_inst) + '\n')
            return

        if arg is None:
            method()
        else:
            method(arg)

    def get_method(self, method_name, arg=None):
        '''
        Wraps the call_method() and the arguments 'method_name' and 'arg' to
        return it as a argument-less function object.

        The reason is that PyQt cannot connect to any method with arguments.
        '''

        # Find out if the method_name is present in the core object.
        # If not, return None.
        try:
            getattr(self.core, method_name)

        except Exception as exception_inst:
            print("The controller tries to get method '{}' in 'Core' object,".format(method_name))
            print("but " + str(exception_inst) + '\n')
            return None

        return partial(self.call_method, method_name, arg)



class MockController(object):
    def call_method(self, method_name, arg=None):
        if arg is None:
            print('Core method {}() called'.format(str(method_name)))
        else:
            print('Core method {}() called with arg = {}'.format(str(method_name), str(arg)))

    def get_method(self, method_name, arg=None):
        return partial(self.call_method, method_name, arg)




