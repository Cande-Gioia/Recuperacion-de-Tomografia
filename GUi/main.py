from PyQt5 import QtWidgets, uic
from imageWidget import imageWidget
from imageHandler import imageHandler

def reconstructAndDisplay(self):
    handler.reconstruct()
    reconstruccion.setImage(handler.reconstruction) #se podra inicializar con punteros para evitar este paso???
    reconstruccion.displayImage()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    ventana = uic.loadUi("GUi/GUI.ui")
    ventana.show()

    handler = imageHandler()

    imagen = imageWidget(handler.image)
    reconstruccion = imageWidget(handler.reconstruction)
    sinograma = imageWidget(handler.sinogram)
    fft = imageWidget(handler.fft)
    proyeccion = imageWidget()
    filtro = imageWidget()
    nada1 = imageWidget()
    nada2 = imageWidget()

    ventana.imagen.addWidget(imagen.canvas)
    ventana.reconstruccion.addWidget(reconstruccion.canvas)
    ventana.sinograma.addWidget(sinograma.canvas)
    ventana.fft.addWidget(fft.canvas)
    ventana.proyeccion.addWidget(proyeccion.canvas)
    ventana.filtro.addWidget(filtro.canvas)
    ventana.nada1.addWidget(nada1.canvas)
    ventana.nada2.addWidget(nada2.canvas)

    ventana.abrir.clicked.connect(imagen.displayImage)
    ventana.abrir.clicked.connect(sinograma.displayImage)
    ventana.abrir.clicked.connect(fft.displayImage)

    ventana.comboBoxAlg.currentIndexChanged.connect(handler.setAlgIndex)
    ventana.comboBoxFilt.currentIndexChanged.connect(handler.setFiltIndex)
    ventana.reconstruir.clicked.connect(reconstructAndDisplay)


    app.exec()


