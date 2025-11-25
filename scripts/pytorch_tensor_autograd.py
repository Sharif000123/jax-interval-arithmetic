import torch
from torchvision.models import resnet18, ResNet18_Weights
# print("Intel",torch.xpu.is_available())
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1,1000)

prediction = model(data)

loss = (prediction - labels).sum()
loss.backward()
# print(loss.retain_grad())

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=9e-1)
optim.step() #gradient descent / absteigendes m (Steigungswert)

print(optim)