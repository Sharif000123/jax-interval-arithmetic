import torch
import numpy as np


data = [[1, 2], [3, 4]]
xData = torch.tensor(data)

npArray = np.array(data)
xNp = torch.from_numpy(npArray)

xOnes = torch.ones_like(xData)
# print(f"Ones Tensor: \n {xOnes} \n")

xRand = torch.rand_like(xData, dtype=torch.float)
# print(f"Random Tensor: \n {xRand} \n")

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor} \n")

# tensor = torch.rand(3, 4) # matrix of three rows and four columns
# if torch.cuda.is_available():
#     tensor = tensor.to('cuda') #moving tensor to GPU
#     print("Tensor moved to GPU")
# print(f"Tensor: {tensor}") # printing the matrix/tensor
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

tensor = torch.ones(4, 4)
# print(tensor)
tensor[:, 1] = 0
# print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# print(f"tensor * tensor \n {tensor * tensor}")
# print(f"tensor * tensor \n {tensor * tensor}")

# print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# # Or
# print(f"tensor @ tensor.T \n {tensor @ tensor.T} \n")

print("----")

# print(tensor, "\n")
# tensor.add_(5) # adds value chosen to all values of the tensor
# print(tensor)

t = torch.ones(5)  # number of values in tensor (size)
# t= t.add_(1)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t = t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")