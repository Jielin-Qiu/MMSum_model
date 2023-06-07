import torch

#If CUDA is available, prints True. Else, False.
print(torch.cuda.is_available())

# If CUDA is availalbe, prints integer of CUDA devices that are available.
print(torch.cuda.device_count())

# If CUDA is available, prints the ID of the current device that is ready for usage.
print(torch.cuda.current_device())
ID = torch.cuda.current_device()

# If CUDA is available, instantiate device you will be using.
device = torch.device(f'cuda:{ID}')
print(device)

# If CUDA is available, get the name of the device.
print(torch.cuda.get_device_name(ID))

# If CUDA is available, move or create tensor using device
# Move
x = torch.tensor(1).to(device)
print(x)

# Create
y = torch.tensor(1, device = device)
print(y)

# If x or y printed "tensor(1, device='cuda:0')" you should be good to go!