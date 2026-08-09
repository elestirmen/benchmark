import tensorflow as tf
import time

print('TensorFlow version:', tf.__version__)
print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))

if tf.config.list_physical_devices('GPU'):
    print('CUDA GPU is accessible!')
    # Perform a dummy tensor operation to trigger any JIT if it existed
    t0 = time.time()
    tf.random.normal([1000, 1000]) @ tf.random.normal([1000, 1000])
    t1 = time.time()
    print(f'Dummy matrix multiplication took {t1-t0:.4f} seconds.')
    print('If this was extremely fast, then there is NO JIT compilation issue with Compute Capability 12.0!')
else:
    print('GPU NOT FOUND!')
