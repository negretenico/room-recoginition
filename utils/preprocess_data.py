def preprocess_new_images(images, labels, label_encoder):
    # Preprocess images (e.g., normalize pixel values to [0, 1])
    images = images / 255.0

    # Convert string l;abels to numerical representations using the same label encoder
    encoded_labels = label_encoder.transform(labels)

    return images, encoded_labels
