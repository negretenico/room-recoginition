from sklearn.model_selection import train_test_split


def split_data(images, labels, label_encoder, test_size=0.2, random_state=42, ):
    # Preprocess images (e.g., normalize pixel values to [0, 1])
    images = images / 255.0
    # Label encoding may not be necessary if your labels are already integers
    # Ensure that labels are in the required format for your loss function
    # Convert string labels to numerical representations
    encoded_labels = label_encoder.fit_transform(labels)
    return train_test_split(images, encoded_labels, test_size=test_size, random_state=random_state)
