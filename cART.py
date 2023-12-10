import numpy as np
from skimage.transform import rotate, radon
from skimage.io import imread, imshow
from matplotlib import pyplot as plt

def cART(sg, theta, niter): 

    step = 1

    angles = np.arange(0, theta, 1)

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

    return np.array(ir)




if __name__ == '__main__':
    img = imread('Screenshot from 2023-12-06 18-55-46.png')
    imshow(img)
    plt.show()
    rad = radon(img[...,0])
    print(type(rad))
    imshow(rad)
    plt.show()
    rec = cART(np.matrix(rad), 180, 1)
    print(type(rec))
    imshow(rec)
    plt.show()