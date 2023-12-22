import os

import cv2
import numpy as np

BASE_IMAGE_SHAPE = (244, 244)


def load_images_from_directory(directory):
    images = []
    labels = []

    for index, label in enumerate(os.listdir(directory)):
        label_path = os.path.join(directory, label)

        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, BASE_IMAGE_SHAPE)

                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)
