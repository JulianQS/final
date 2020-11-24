'''           Pontificia Universidad Javeriana
              Procesamiento de imagenes y visión
                  Alejandra Avendaño Cortina
                   Parcial Final 24/11/2
'''

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time
from bandera import *

if __name__ == '__main__':

    path = 'C:/Users/aleja/OneDrive/Documentos/Procesamiento/Talleres/Preparcial/imagenes'
    image_name = 'flag1.png'
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    cv2.waitKey(0)
    arch = bandera()  # The system call colorImage
    arch.porcentaje()  # The system call the function