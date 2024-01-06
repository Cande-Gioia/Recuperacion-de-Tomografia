from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib import pyplot as plt
import matplotlib as mpl

class displayWidget():
    def __init__(self):
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot()
        mpl.rc('image', cmap='gray')


class imageWidget(displayWidget):
    def __init__(self, image=0):
        print('se inicio una imagen')
        super().__init__()
        self.image = image

    def displayImage(self):   
        self.ax.imshow(self.image)
        self.canvas.draw()

    def setImage(self, image):
        self.image = image


class plotWidget(displayWidget):
    def __init__(self, data=0):
        print('se inicio un gráfico')
        super().__init__()

