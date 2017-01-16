__version__ = '1.0'

from model import *

def launch():
    app = QtGui.QApplication(sys.argv)
    core = Core()
    sys.exit(app.exec_())

if __name__ == '__main__':
    launch()
