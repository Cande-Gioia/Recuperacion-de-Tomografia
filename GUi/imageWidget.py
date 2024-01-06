from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib import pyplot as plt

class imageWidget():
    def __init__(self):
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)