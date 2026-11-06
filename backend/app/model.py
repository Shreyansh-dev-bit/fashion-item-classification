import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_fashion_mnist_cnn(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Constructs the canonical Fashion-MNIST CNN architecture:
    1. Input layer (28, 28, 1)
    2. Conv2D (32 filters, 3x3 kernel, ReLU)
    3. MaxPooling2D (2x2 pool size)
    4. Conv2D (64 filters, 3x3 kernel, ReLU)
    5. MaxPooling2D (2x2 pool size)
    6. Flatten
    7. Dense (128 units, ReLU)
    8. Dropout (rate 0.3)
    9. Dense (10 units, Softmax)
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape, name="input_image"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv2d_1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense_features"),
            layers.Dropout(dropout_rate, name="dropout"),
            layers.Dense(num_classes, activation="softmax", name="output_probabilities"),
        ],
        name="FashionMNIST_CNN",
    )
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Compiles the CNN model with Adam optimizer, sparse categorical crossentropy loss,
    and classification accuracy metric.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.SparseCategoricalCrossentropy()
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
