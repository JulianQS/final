import numpy as np
import os
import cv2
import sys
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time #import time library
from recreate_image import *
from hough import hough

class bandera:

    def __init__(self): #Initialize the class

        path = 'C:/Users/aleja/OneDrive/Documentos/Procesamiento/Talleres/Preparcial/imagenes'
        image_name = 'flag1.png'
        path_file = os.path.join(path, image_name)
        image = cv2.imread(path_file)
        cv2.waitKey(0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        image = np.array(image, dtype=np.float64) / 255  # Normalize data from 0 to 1


    def color(self):

        for n_cluster in range(10):  # segment with 1 to 10 clusters
            print('Clustering for ', n_cluster + 1, 'clusters\n')  # print string
            n_colors = 4

        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))  # Reorder data
        t0 = time()  # Save actual time
        image_array_sample = shuffle(image_array, random_state=0)[:10000]  # Sample original image
        model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)  # define the model

        print("Done in %0.3fs." % (time() - t0))  # print time
        # Get labels for all points
        t0 = time()  # Save actual time

        print("Predicting color indices on the full image (Kmeans)")  # print string
        labels = model.predict(image_array)  # Assign a label to each pixel
        print("Done in %0.3fs." % (time() - t0))  # print time

        set_labels = set(labels)
        len(labels)
        print(len(set_labels))

    def porcentaje(self):

        vector = [0, 0, 0, 0]

        for i in labels:
            if i == 1:
                vector[0] = vector[0] + 1
            elif i == 2:
                vector[1] = vector[1] + 1
            elif i == 3:
                vector[2] = vector[2] + 1
            else:
                vector[3] = vector[3] + 1

        # Porcentajes
        porc0 = (vector[0] / len(labels)) * 100

        if (porc0 != 0):
            print(porc0)

        porc1 = (vector[1] / len(labels)) * 100
        if (porc1 != 0):
            print(porc1)

        porc2 = (vector[2] / len(labels)) * 100
        if (porc2 != 0):
            print(porc2)

        porc3 = (vector[3] / len(labels)) * 100
        if (porc3 != 0):
            print(porc3)

        Porcentajes = [porc0, porc1, porc2, porc3]