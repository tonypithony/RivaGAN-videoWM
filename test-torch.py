# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import torch

print(torch.cuda.is_available())
# True
print(torch.cuda.device_count())
# 1
print(torch.cuda.current_device())
# 0
print(torch.cuda.device(0))
# <torch.cuda.device object at 0x721fb5ab4130>
print(torch.cuda.get_device_name(0))
# 'NVIDIA GeForce RTX 4060 Ti'
