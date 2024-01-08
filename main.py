from PyQt5 import QtWidgets, uic

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    ventana = uic.loadUi("GUI.ui")
    ventana.show()


    app.exec()

