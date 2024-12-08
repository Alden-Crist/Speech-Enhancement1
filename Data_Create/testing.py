import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Is GPU available:", tf.config.list_physical_devices('GPU'))
import tensorflow as tf

# List all physical devices recognized by TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)
