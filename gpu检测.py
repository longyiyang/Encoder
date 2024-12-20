import torch

# 判断是否有可用的 GPU
gpu_available = torch.cuda.is_available()

if gpu_available:
    print("GPU is available!")
else:
    print("GPU is not available, using CPU.")
