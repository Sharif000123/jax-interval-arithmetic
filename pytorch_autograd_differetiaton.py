import torch

# a = torch.tensor([2.,3.], requires_grad=True)
# b = torch.tensor([6.,4.], requires_grad=True)
# # x = torch.tensor[6.,4.]
# # print(x)

# Q = 3*a**3 - b**2

# externalGrad = torch.tensor([1.,1.])
# Q.backward(gradient=externalGrad)

# print(9*a**2 == a.grad)
# print(-2*b == b.grad)

# Exclusion from the DAG

# x = torch.rand(5,5)
# y = torch.rand(5,5)
# z = torch.rand((5,5), requires_grad=True)

# a = x + y
# b = x + z

# print(f"Does `a` require gradients? {a.requires_grad}")
# print(f"Does `b` require gradients? {b.requires_grad}")

#2. Task
from torch import nn, optim
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad=False


model.fc = nn.Linear(512, 10)

optimizer = optim.SGD(model.parameters(), lr=0.02,momentum=0.9)

# torch.no_grad()
