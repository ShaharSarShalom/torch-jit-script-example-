import torchvision
import torch
from time import perf_counter
import numpy as np


def timer(f, *args):
    start = perf_counter()
    f(*args)
    return (1000 * (perf_counter() - start))


# Example 1.1 Pytorch cpu version

model_ft = torchvision.models.resnet18(pretrained=True)
model_ft.eval()
x_ft = torch.rand(1, 3, 224, 224)

# warm up the model https://github.com/pytorch/pytorch/issues/34703
for _ in range(10):
    model_ft(x_ft)

torch_cpu_sec = np.mean([timer(model_ft, x_ft) for _ in range(10)])

# Example 2.1 torch.jit.script cpu version

# script_cell = torch.jit.script(model_ft, (x_ft))
script_cell = torch.jit.script(model_ft)
script_cpu_sec = np.mean([timer(script_cell, x_ft) for _ in range(10)])

# Example 1.2 Pytorch gpu version
model_ft_gpu = torchvision.models.resnet18(pretrained=True).cuda()
if True:
    model_ft_gpu.half()
    x_ft = x_ft.half()

x_ft_gpu = x_ft.cuda()
model_ft_gpu.eval()
torch_gpu_sec = np.mean([timer(model_ft_gpu, x_ft_gpu) for _ in range(100)])


# Example 2.2 torch.jit.script gpu version

#script_cell_gpu = torch.jit.script(model_ft_gpu, (x_ft_gpu))
script_cell_gpu = torch.jit.script(model_ft_gpu)
script_gpu_sec = np.mean([timer(script_cell_gpu, x_ft.cuda()) for _ in range(100)])

print('cpu torch {}, cpu script {}'.format(torch_cpu_sec, script_cpu_sec))
print('gpu torch {}, gpu script {}'.format(torch_gpu_sec, script_gpu_sec))