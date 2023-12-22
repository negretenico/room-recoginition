from keras import models
from keras.src.layers import BatchNormalization
from tensorflow.python.layers import layers


class AdvancedCNN(models.Model):
    def __init__(self, input_shape, num_classes):
        super(AdvancedCNN, self).__init__()

        # Convolutional Block 1
        self.conv1_1 = layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape)
        self.conv1_2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.batch_norm1 = BatchNormalization()
        self.max_pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))
        self.dropout1 = layers.Dropout(0.25)

        # Convolutional Block 2
        self.conv2_1 = layers.Conv2D(128, (3, 3), activation='relu')
        self.conv2_2 = layers.Conv2D(128, (3, 3), activation='relu')
        self.batch_norm2 = BatchNormalization()
        self.max_pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))
        self.dropout2 = layers.Dropout(0.25)

        # Convolutional Block 3
        self.conv3_1 = layers.Conv2D(256, (3, 3), activation='relu')
        self.conv3_2 = layers.Conv2D(256, (3, 3), activation='relu')
        self.batch_norm3 = BatchNormalization()
        self.max_pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))
        self.dropout3 = layers.Dropout(0.25)

        # Flatten and Dense Layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout4 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None):
        # Forward pass
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.batch_norm1(x)
        x = self.max_pool1(x)
        x = self.dropout1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.batch_norm2(x)
        x = self.max_pool2(x)
        x = self.dropout2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.batch_norm3(x)
        x = self.max_pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout4(x)
        x = self.dense2(x)

        return x
