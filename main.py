import logging
import os.path
from typing import Final

import numpy as np
from sklearn.preprocessing import LabelEncoder

from data import read
from data import split
from utils import make_model, train_model
from utils.decode_label import decode_label
from utils.preprocess_data import preprocess_new_images

NUM_CHANNELS: Final = 3
BATCH_SIZE: Final = 1
NUM_CLASSES: Final = 5
INPUT_SHAPE: Final = (244, 244, 3)
label_encoder = LabelEncoder()
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
    logging.info("STARTING APP")
    logging.info("Reading data in")
    images, labels = read.load_images_from_directory(os.path.join('multiclass'))
    logging.info(f"""
    Done reading in data
    we have {len(labels)} labels and {len(images)} images 
""")
    train_images, test_images, train_labels, test_labels = split.split_data(images=images, labels=labels,
                                                                            label_encoder=label_encoder)
    logging.info("Creating the model")
    model = make_model.make_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    logging.info("Done creating the model")
    logging.info("Staring the model training process")
    train_model.train_vit_model(model, (train_images, train_labels), (test_images, test_labels))
    logging.info("Done with the model training process")
    logging.info('Reading in test data')
    t_images, t_labels = read.load_images_from_directory(os.path.join('test_data'))
    new_t_images, new_t_encoded_labels = preprocess_new_images(t_images, t_labels, label_encoder)
    predictions = model.predict(new_t_images)
    total = 0
    for i in range(len(new_t_images)):
        image = new_t_images[i]
        encoded_label = new_t_encoded_labels[i]
        current_prediction = predictions[i]
        decoded_label = decode_label(encoded_label, label_encoder)
        predicted_category = decode_label(np.argmax(current_prediction), label_encoder)
        print(f"Image {i + 1}: Actual Label: {decoded_label}, Predicted Category: {predicted_category}")
        total += decoded_label == predicted_category
    logging.info(f"We looked at {len(new_t_images)} images and we correctly guess {total}")
