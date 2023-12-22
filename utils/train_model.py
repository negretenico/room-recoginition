import tensorflow as tf


def train_vit_model(model, train_data, val_data, epochs=1) -> None:
    # Unpack the train and validation data
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    # Train the model
    with tf.device('/GPU:0'):
        model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=epochs)
