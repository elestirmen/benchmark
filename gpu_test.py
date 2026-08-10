import tensorflow as tf
import time

print('TensorFlow version:', tf.__version__)
print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))

if tf.config.list_physical_devices('GPU'):
    print('CUDA GPU is accessible!')
    # This probes one simple TensorFlow operation only. Individual model kernels
    # can still trigger separate PTX/JIT compilation during real inference.
    t0 = time.time()
    tf.random.normal([1000, 1000]) @ tf.random.normal([1000, 1000])
    t1 = time.time()
    print(f'Dummy matrix multiplication took {t1-t0:.4f} seconds.')
    print('This confirms only this matrix operation. Run a real model smoke test to validate model-specific kernels/JIT.')
else:
    print('GPU NOT FOUND!')
