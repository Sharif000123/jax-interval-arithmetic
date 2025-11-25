import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR  


# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

import numpy as np
import matplotlib.pyplot as plt

# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    
# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

#show images
# imshow(torchvision.utils.make_grid(images))
#print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        #fc is a fully connected layer, thats also why Linear() is used
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # Linear(input_size, output_size)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self, x):
        x = self.pool(self.prelu1(self.conv1(x)))
        x = self.pool(self.prelu2(self.conv2(x)))
        x = torch.flatten(x,1)
        x = self.prelu3(self.fc1(x))
        x = self.prelu4(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.75)
scheduler = StepLR(optimizer, step_size=2000, gamma=0.5) # Reduce lr by 0.1 every ... epochs



for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): # enumerate() is a function to keep track of the number of iterations in a loop
        # getting inputs, data is a list of [input, labels]
        input, labels = data
        # zero parameter gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(input)
        loss = criterion(output, labels) # calculating sqared mean error/difference between output and input (values)
        loss.backward()
        optimizer.step()
        if i % 1000 == 999:
            for param_group in optimizer.param_groups:  
                if (param_group['momentum'] * 1.001) <= 0.99:
                    param_group['momentum'] *= 1.001
                    print(f'New momentum: {param_group["momentum"]}, learning rate: {param_group["lr"]}')
        
        # print stats
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    scheduler.step() #reducing learn rate

print('Finished training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('Ground Truth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

output = net(images)

_, predicted = torch.max(output, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total}%')

# count predictions per class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        _, predictions = torch.max(output, 1)
        # collecting prediction for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


for classname, correct_count in  correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

# net.to(device)