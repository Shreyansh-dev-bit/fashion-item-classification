import os
import random
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from app.config import FASHION_MNIST_CLASSES


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_preprocess_data(
    val_size: int = 6000,
    random_seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load real Fashion-MNIST dataset using tf.keras.datasets.fashion_mnist.load_data().
    Splits 60,000 training samples into 54,000 train and 6,000 validation.
    Holds out official 10,000 test samples.
    Normalizes pixel values to [0.0, 1.0] and reshapes to (28, 28, 1).
    """
    set_seed(random_seed)
    
    print("[Dataset] Loading Fashion-MNIST via tensorflow.keras.datasets.fashion_mnist.load_data()...")
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    total_train_samples = len(x_train_full)
    if total_train_samples != 60000:
        raise ValueError(f"Expected 60,000 training samples from Fashion-MNIST, got {total_train_samples}")
    if len(x_test) != 10000:
        raise ValueError(f"Expected 10,000 test samples from Fashion-MNIST, got {len(x_test)}")

    # Split into 54,000 train and 6,000 validation using stratified split
    train_size = total_train_samples - val_size
    print(f"[Dataset] Splitting {total_train_samples} samples into {train_size} train and {val_size} validation (stratified)...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_seed,
        stratify=y_train_full
    )
    
    # Normalize pixel values from 0-255 to 0.0-1.0 (float32)
    print("[Dataset] Normalizing pixel intensities to [0.0, 1.0] and reshaping to (28, 28, 1)...")
    x_train = (x_train.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
    x_val = (x_val.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
    x_test = (x_test.astype("float32") / 255.0).reshape(-1, 28, 28, 1)

    # Ensure labels are 1D int arrays
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    print(f"[Dataset] Data ready: Train={x_train.shape}, Val={x_val.shape}, Test={x_test.shape}")
    print(f"[Dataset] Target classes: {len(FASHION_MNIST_CLASSES)} categories ({', '.join(FASHION_MNIST_CLASSES[:4])}...)")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
