import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0))
print("PyTorch CUDA version:", torch.version.cuda)

import tensorflow as tf
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
