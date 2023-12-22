import tensorflow as tf
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy

from models.model import AdvancedCNN


def make_model(num_classes, input_shape, has_logits=True):
    model = AdvancedCNN(input_shape=input_shape,
                        num_classes=num_classes)

    # Define the loss function, optimizer, and metrics
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    accuracy_metric = SparseCategoricalAccuracy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy_metric])

    return model
