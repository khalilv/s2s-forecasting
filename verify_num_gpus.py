import torch
print('GPU is available: ', torch.cuda.is_available())
print('Number of GPUs available: ', torch.cuda.device_count())