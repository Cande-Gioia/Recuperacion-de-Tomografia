import numpy as np
from skimage.transform import rotate, radon
from skimage.transform import rescale
from skimage.data import shepp_logan_phantom
from scipy.fft import fft, fftshift, ifft, ifftshift, fft2

class imageHandler():

    def __init__(self):
        #Para probar
        image = shepp_logan_phantom()
        self.image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)

        self.maxAngle = 180
        self.n = 180
        self.proyectionAngles = np.linspace(0.0, self.maxAngle, self.n+1)

        self.reconstruction = None
        self.sinogram = radon(self.image, theta=self.proyectionAngles)
        self.fft = np.abs(fft2(self.image))
        self.filter = None
        self.proyection = None

        self.algIndex = 0
        self.filtIndex = 0

        
    def setMaxAngle(self, angle):
        self.maxAngle = angle
        self.proyectionAngles = np.linspace(0.0, self.maxAngle, self.n+1)

    def setProyectionAmount(self, n):
        self.n = n
        self.proyectionAngles = np.linspace(0.0, self.maxAngle, self.n)
   
    def makeSinogram(self):
        self.sinogram = radon(self.image, theta=self.proyectionAngles)

    def setAlgIndex(self, index):
        self.algIndex = index

    def setFiltIndex(self, index):
        self.filtIndex = index

    def reconstruct(self):
        if self.algIndex == 0:
            self.cART()

    def cART(self): #use esta version del algoritmo porque el resultado es un poco mejor

        step = 1
        niter = 1

        angles = self.proyectionAngles
        sg = np.matrix(self.sinogram)
        theta = self.maxAngle


        ir = np.matrix(np.zeros((len(sg), len(sg))))
        tdiff = np.matrix(np.zeros((len(sg), len(sg))))

        for i in range(niter):
            for j in range(len(angles)):

                ir = rotate(ir, step, order=3)
                tmp = np.matrix(np.sum(np.repeat(sg[:,j], len(ir), axis=1), axis=1))
                temp = np.matrix(np.sum(ir, axis=1)).T
                diff = (tmp - temp) / len(ir)
                
                for k in range(len(ir)):
                    tdiff[:, k] = diff

                ir = ir + tdiff

        self.reconstruction = np.array(ir)