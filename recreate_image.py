import numpy as np
import cv2
import os


def recreate_image (centers, labels, rows, cols):#create a function
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d)) #create zeros matrix
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters