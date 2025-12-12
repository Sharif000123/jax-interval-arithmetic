import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input)) #convolution check overlaps of the data, or shared integrals between the data/of the data
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2,2)) #max_pool is choosing the biggest number in a certain area
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1) #flattening is extracting the data from the x-dimensional tensors and saving the values in a single line (1 dimensional tensor)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4)) #relu just takes the input and exchanges every negative value with a 0
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output
    
net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1,1,32,32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1,10))

output = net(input)
target = torch.randn(10)
# print(target)
target = target.view(1, -1)
criterion = nn.MSELoss() # the MSELoss function calculates the mean squared error between the output and the target values, in other words it calculates the delta between the output and target values and squares each delta and then calculates the mean of all the squared deltas

loss = criterion(output, target)
print(loss)

print(loss.grad_fn) #MSELoss
print(loss.grad_fn.next_functions[0][0]) #Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #ReLU

net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=learning_rate)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target) # calculates the squared mean error/difference/delta of the output and target values
loss.backward() # calculates the error of the output and target values, starting from the output layer, and then backtracking through the layers of the network to the input layer

optimizer.step() # updates the weights of the based on the error calculated by the loss.backward() function
