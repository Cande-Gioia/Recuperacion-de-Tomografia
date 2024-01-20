from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QListWidgetItem, QColorDialog, QFileDialog, QDialog, QStyle
from PyQt5.QtCore import Qt, QCoreApplication

import numpy as np
import matplotlib.pyplot as plt 
from skimage import color, io, util, transform
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

from pathlib import Path

import matplotlib as mpl
mpl.rc('image', cmap='gray') #para no tener que pasar cmap='gray' cada vez que ploteo algo

from scipy import fft
from scipy.fft import fftfreq, fftshift, ifft, ifftshift, fft, fft2
from scipy import sparse
from scipy.signal import get_window
from scipy import optimize

app = QtWidgets.QApplication([])
ventana = uic.loadUi("GUI.ui")



def radon(im, P):
    angles = np.arange(0, 180, 180/P)
    return transform.radon(im, theta=angles)

def ri_python(s):
    N = s.shape[0]
    P = s.shape[1]
    angles = np.arange(0, 180, 180/P)

    return transform.iradon(s, theta=angles)

def ri_ART(s):
    N = s.shape[0]
    P = s.shape[1]
    angles = np.arange(0, 180, 180/P)

    ir = np.zeros((N, N))
    step = angles[1] - angles[0]

    niter = 1
    for i in range(niter):
        for j in reversed(range(P)):
            ir = transform.rotate(ir, step, order=5)
            tmp = s[:,j] * N
            temp = np.sum(ir, axis=0)
            diff = (tmp - temp) / N
            ir = ir + np.tile(diff, (N, 1))

    return ir

def make_filter(filtro, N): 
    freq_resp = abs(np.linspace(-1,1,N))

    if filtro == 'Ram-Lak':
        freq_resp = freq_resp

    elif filtro == 'Shepp-Logan':
        freq_resp = freq_resp * np.sinc(freq_resp/2)

    elif filtro == 'Coseno':
        freq_resp = freq_resp * get_window('cosine', len(freq_resp))

    elif filtro == 'Hamming':
        freq_resp = freq_resp * get_window('hamming', len(freq_resp))

    elif filtro == 'Hann':
        freq_resp = freq_resp * get_window('hann', len(freq_resp))

    else:
        freq_resp = 1

    return freq_resp


def ri_FBP(s):
    N = s.shape[0]
    P = s.shape[1]
    angles = np.arange(0, 180, 180/P)

    bp = np.zeros((N,N))
    tmp = bp.copy()

    i_filtro = ventana.cb_filtro.currentText()

    filt = make_filter(i_filtro, N)

    sinog = np.swapaxes(s, 0, 1)
    fft_R = fftshift(fft(sinog))
    filtproj = ifftshift(fft_R * filt)
    s = np.real(ifft(filtproj))
    s = np.swapaxes(s, 0, 1)


    for i in range(P):
        tmp = np.tile(s[:,i], (N, 1))
        tmp = transform.rotate(tmp, angles[i], order=1, clip=True)
        bp = bp + tmp

    return bp

def ri_fourier(s):
    N = s.shape[0]
    P = s.shape[1]
    angles = np.arange(0, 180, 180/P)

    return s

def ri_GC(s):
    N = s.shape[0]
    P = s.shape[1]
    angles = np.arange(0, 180, 180/P)

    pi = np.pi
    cos = np.cos
    sin = np.sin

    rot_mat = []
    for ang in angles:
        rot_mat.append(np.array([[cos(ang*pi/180), -sin(ang*pi/180)], 
                                 [sin(ang*pi/180), cos(ang*pi/180)]], 
                                 dtype=np.float32))

    col_ptr = np.empty(N*N + 1, dtype=np.int32)
    q = 0
    col_ptr[0] = 0
    row_idx = []
    vals = []

    off = (N//2 - 0.5) if N % 2 == 0 else N//2
    X = Y = np.linspace(-off, +off, N)

    a = 1

    for yi in Y:
        for xi in X:
            for r, rot in enumerate(rot_mat):
                v = (rot @ [xi,yi])

                if v[0] < -off or v[0] > off or v[1] < -off or v[1] > off:
                    continue

                x = v[0]+off
                y = off-v[1]
                xint = int(x)
                yint = int(y)
                xm = x - xint
                ym = y - yint

                if xm < 0.001:
                    row_idx.append(xint*P + r)
                    vals.append(1)
                    q += 1
                elif xm > 0.999:
                    row_idx.append((xint+1)*P + r)
                    vals.append(1)
                    q += 1
                else:
                    row_idx.extend([xint*P + r, (xint+1)*P + r])
                    vals.extend([(1-xm),xm])
                    q += 2
            
            col_ptr[a] = q
            a += 1

    A = sparse.csc_array((vals, row_idx, col_ptr), shape=(N*P, N*N)) #esta es la matriz, notar que es dispersa (sino no alcanza la memoria)
    b = s.flatten()

    # con la matriz A generada en el paso previo, y con la entrada que es b (sinograma),
    # aproximo la inversa de la transformación
    # para eso, minimizo el campo f(x) = ||Ax-b||
    AT = A.T
    ATB =  AT @ b

    f = lambda x: (A @ x - b).T @ (A @ x - b)
    gradf = lambda x: 2*((AT @ (A @ x)) - ATB)

    x0 = np.zeros(N*N, dtype=np.float32)

    res = optimize.minimize(f, x0, method='Newton-CG', jac=gradf, options={'maxiter':10})

    return res.x.reshape(N, N)

radon_inv_callbacks = [ri_python, ri_ART, ri_FBP, ri_fourier, ri_GC]

def imagen_desde_archivo():
    options = QFileDialog.Options()
    # options |= QFileDialog.DontUseNativeDialog
    f, _ = QFileDialog.getOpenFileName(ventana, "Select files", "","All Files (*);;JPG files (*.jpg);;PNG files (*.png)", options=options)

    if f == '':
        return None, ''

    input = io.imread(f)
    image = util.img_as_float32(transform.resize(input, (256, 256), anti_aliasing=True))[...,0]
    assert len(image.shape) == 2 and image.shape[0] == image.shape[1]

    return image, f

def seleccionar_proyeccion():
    ventana.w_proy.canvas.ax.clear()
    row = ventana.lw_sinogramas.currentRow()
    if row < 0:
        ventana.w_proy.canvas.draw()
        return

    sino = ventana.lw_sinogramas.item(row).data(Qt.UserRole)
    j = ventana.sp_proy2.value()
    if j >= sino.shape[1]:
        ventana.w_proy.canvas.draw()
        return
    
    ventana.w_proy.canvas.ax.plot(sino[:,j])
    ventana.w_proy.canvas.ax.grid()
    ventana.w_proy.canvas.draw()

def seleccionar_proyeccion_filtrada():
    ventana.w_proy_filt.canvas.ax.clear()
    row = ventana.lw_sinogramas.currentRow()
    if row < 0:
        ventana.w_proy_filt.canvas.draw()
        return

    sino = ventana.lw_sinogramas.item(row).data(Qt.UserRole)
    j = ventana.sp_proy_filt.value()
    if j >= sino.shape[1]:
        ventana.w_proy_filt.canvas.draw()
        return

    s = sino[:,j]

    i_filtro = ventana.cb_filtro_2.currentText()

    filt = make_filter(i_filtro, len(s))
 
    fft_R = fftshift(fft(s))
    filtproj = ifftshift(fft_R * filt)
    sg = np.real(ifft(filtproj))

    ventana.w_proy_filt.canvas.ax.plot(sg)
    ventana.w_proy_filt.canvas.ax.grid()
    ventana.w_proy_filt.canvas.draw()

def seleccionar_filtro(val):
    filt = make_filter(val, 256)

    ventana.w_filtro.canvas.ax.clear()
    ventana.w_filtro.canvas.ax.plot(filt)
    ventana.w_filtro.canvas.ax.grid()
    ventana.w_filtro.canvas.draw()

    seleccionar_proyeccion_filtrada()

def seleccionar_im_entrada():
    row = ventana.lw_entrada.currentRow()
    if row >= 0:
        image = ventana.lw_entrada.item(row).data(Qt.UserRole)
        ventana.w_entrada.canvas.ax.imshow(image)
    else:
        ventana.w_entrada.canvas.ax.clear()
    ventana.w_entrada.canvas.draw()

def seleccionar_sinograma():
    row = ventana.lw_sinogramas.currentRow()
    if row >= 0:
        sino = ventana.lw_sinogramas.item(row).data(Qt.UserRole)
        ventana.w_sinograma.canvas.ax.imshow(sino)
    else:
        ventana.w_sinograma.canvas.ax.clear()
    ventana.w_sinograma.canvas.draw()
    seleccionar_proyeccion()
    seleccionar_proyeccion_filtrada()


def seleccionar_im_salida():    
    row = ventana.lw_salida.currentRow()
    if row >= 0:
        image = ventana.lw_salida.item(row).data(Qt.UserRole)
        ventana.w_salida.canvas.ax.imshow(image)
    else:
        ventana.w_salida.canvas.ax.clear()
    ventana.w_salida.canvas.draw()

def fft_disp():
    row = ventana.lw_entrada.currentRow()
    if row < 0:
        return

    if row >= 0:
        image = ventana.lw_entrada.item(row).data(Qt.UserRole)
        transform = fftshift(fft2(ifftshift(image)))
        ventana.w_fft.canvas.ax.imshow(np.abs(transform))
    else:
        ventana.w_fft.canvas.ax.clear()
    ventana.w_fft.canvas.draw()

def btn_agregar_entrada_cb():
    image, f = imagen_desde_archivo()
    #if not image:  #comentado porque si no crashea
    #    return

    qlwt = QListWidgetItem()
    qlwt.setData(Qt.UserRole, image)
    qlwt.setText(Path(f).stem)
    ventana.lw_entrada.addItem(qlwt)
    ventana.lw_entrada.setCurrentRow(ventana.lw_entrada.count() - 1)

    seleccionar_im_entrada()
    fft_disp()

def btn_agregar_sinograma_cb(): 
    image, f = imagen_desde_archivo()    
    if not image:
        return

    qlwt = QListWidgetItem()
    qlwt.setData(Qt.UserRole, image)
    qlwt.setText(Path(f).stem)
    ventana.lw_sinogramas.addItem(qlwt)
    ventana.lw_sinogramas.setCurrentRow(ventana.lw_sinogramas.count() - 1)

    seleccionar_sinograma()

def btn_eliminar_entrada_cb():
    ventana.lw_entrada.takeItem(ventana.lw_entrada.currentRow())
    ventana.lw_entrada.setCurrentRow(ventana.lw_entrada.count() - 1)

def btn_eliminar_salida_cb():
    ventana.lw_salida.takeItem(ventana.lw_sinogramas.currentRow())
    ventana.lw_salida.setCurrentRow(ventana.lw_sinogramas.count() - 1)

def btn_eliminar_sinograma_cb():
    ventana.lw_sinogramas.takeItem(ventana.lw_sinogramas.currentRow())
    ventana.lw_sinogramas.setCurrentRow(ventana.lw_sinogramas.count() - 1)

def btn_radon_cb():
    row = ventana.lw_entrada.currentRow()
    if row < 0:
        return

    image = ventana.lw_entrada.item(row).data(Qt.UserRole)
    P = ventana.sp_proy1.value()

    sino = radon(image, P)

    qlwt = QListWidgetItem()
    qlwt.setData(Qt.UserRole, sino)
    qlwt.setText(ventana.lw_entrada.item(row).text()+"_sino")
    ventana.lw_sinogramas.addItem(qlwt)
    ventana.lw_sinogramas.setCurrentRow(ventana.lw_sinogramas.count() - 1)
    seleccionar_sinograma()
    



def btn_radon_inv_cb():
    row = ventana.lw_sinogramas.currentRow()
    if row < 0:
        return
    
    sino = ventana.lw_sinogramas.item(row).data(Qt.UserRole)

    metodo = ventana.cb_metodo.currentIndex()

    reconstruccion = radon_inv_callbacks[metodo](sino)
    nombre = ventana.lw_sinogramas.item(row).text()
    i = nombre.rindex("_")
    if i != -1:
        nombre = nombre[:i]

    qlwt = QListWidgetItem()
    qlwt.setData(Qt.UserRole, reconstruccion)
    qlwt.setText(nombre+"_rec")
    ventana.lw_salida.addItem(qlwt)
    ventana.lw_salida.setCurrentRow(ventana.lw_salida.count() - 1)
    seleccionar_im_salida()


if __name__ == "__main__":
    ventana.sp_proy1.setValue(180)
    ventana.sp_proy2.setValue(1)

    ventana.sp_proy2.valueChanged.connect(seleccionar_proyeccion)
    ventana.sp_proy_filt.valueChanged.connect(seleccionar_proyeccion_filtrada)
    ventana.cb_filtro_2.currentTextChanged.connect(seleccionar_filtro)
    ventana.lw_entrada.currentItemChanged.connect(seleccionar_im_entrada)
    ventana.lw_sinogramas.currentItemChanged.connect(seleccionar_sinograma)
    ventana.lw_salida.currentItemChanged.connect(seleccionar_im_salida)
    ventana.btn_agregar_entrada.clicked.connect(btn_agregar_entrada_cb)
    ventana.btn_eliminar_entrada.clicked.connect(btn_eliminar_entrada_cb)
    ventana.btn_eliminar_salida.clicked.connect(btn_eliminar_salida_cb)
    ventana.btn_agregar_sinograma.clicked.connect(btn_agregar_sinograma_cb)
    ventana.btn_eliminar_sinograma.clicked.connect(btn_eliminar_sinograma_cb)
    ventana.btn_radon.clicked.connect(btn_radon_cb)
    ventana.btn_radon_inv.clicked.connect(btn_radon_inv_cb)
    ventana.btn_radon_inv_2.clicked.connect(btn_radon_inv_cb)

    ventana.show()
    app.exec()

