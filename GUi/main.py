from PyQt5 import QtWidgets, uic
from imageWidget import imageWidget

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    ventana = uic.loadUi("GUi/GUI.ui")
    ventana.show()

    imagen = imageWidget()
    reconstruccion = imageWidget()
    sinograma = imageWidget()
    fft = imageWidget()
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


    app.exec()

