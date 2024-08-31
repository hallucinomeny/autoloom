import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs:", gpus)

# Simple TensorFlow computation
a = tf.constant(2.0)
b = tf.constant(3.0)
c = a + b
print("Computation result:", c.numpy())