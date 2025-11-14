import flax
from flax import nnx
from flax import linen as lnn
# import torch.optim as optim
import flax.jax_utils
import flax.linen
import flax.serialization
import optax
import jax
import jax.numpy as jnp
from flax.training import train_state 
import torchvision #switch to torchvision.datasets instead of tfds, as it does not support my python version
import torchvision.transforms as transforms
# from torch.optim.lr_scheduler import StepLR  


import numpy as np
import matplotlib.pyplot as plt


# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

def preprocess(data):
    image = tf.cast(data['image'], tf.float32) / 255.0
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    return image, data['label']

'''
batch_size = 4

trainset = tfds.load('cifar10', split='train',
                     batch_size=batch_size, as_supervised=True)
# trainloader = tfds.load(tfds.load('cifar10', split='train', batch_size=batch_size, as_supervised=True))

trainset = trainset.map(preprocess).shuffle(10000).batch(batch_size)
trainloader = iter(tfds.as_numpy(trainset))

testset = tfds.load('cifar10', split='test',
                    batch_size=batch_size, as_supervised=True)
testset = testset.map(preprocess).batch(batch_size)
testloader = iter(tfds.as_numpy(testset))
'''

def get_datasets():
    """Load CIFAR-10 train and test datasets into memory."""
    ds_builder = tfds.builder('cifar10')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.array(train_ds['image'], dtype=jnp.float32) / 255.0
    test_ds['image'] = jnp.array(test_ds['image'], dtype=jnp.float32) / 255.0
    return train_ds, test_ds

# testloader = tfds.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_ds)
images, labels = next(dataiter)

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


class Net(flax.linen.Module):

    @flax.linen.compact
    def __call__(self, x):
        x = lnn.Conv(features=6, kernel_size=(5, 5))(x)
        x = lnn.PReLU()(x)
        x = lnn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID') # fills outer space so that all numbers of input are captured
        x = lnn.Conv(features=16, kernel_size=(5, 5))(x)
        x = lnn.PReLU()(x)
        x = lnn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID') # fills outer space so that all numbers of input are captured
        x = x.reshape(x.shape[0], -1)
        x = lnn.Dense(features=120)(x)
        x = lnn.PReLU()(x)
        x = lnn.Dense(features=84)(x)
        x = lnn.PReLU()(x)
        x = lnn.Dense(features=10)(x)
        return x
'''
        # self.conv1 = nn.Conv2d(3,6,5) # 3 is the number of input channels, 6 is the number of output channels, 5 is the kernel size, kernel is the filter that is used to convolve the input image, and the kernel size is the size of the filter
        # self.conv1 = flax.linen.Conv(features=6, kernel_size=(5, 5))(x)
        # # self.pool = nn.MaxPool2d(2,2)
        # self.pool = flax.linen.max_pool(
        #     self, window_shape=(2, 2), strides=(2, 2))
        # # self.conv2 = nn.Conv2d(6,16,5)
        # self.conv2 = flax.linen.Conv(features=16, kernel_size=(5, 5))
        # self.prelu1 = flax.linen.PReLU()(x)
        # self.prelu2 = flax.linen.PReLU()(x)
        # self.prelu3 = flax.linen.PReLU()
        # self.prelu4 = flax.linen.PReLU()
        # # fc is a fully connected layer, thats also why Linear() is used

        # # self.fc1 = nnx.Linear(16 * 5 * 5, 120) # Linear(input_size, output_size)
        # self.fc1 = flax.linen.Dense(120)  # Linear(output_size)
        # self.fc2 = nnx.Linear(84)
        # self.fc3 = nnx.Linear(10)
'''


net = Net()




criterion = lnn.CrossEntropyLoss()
# Reduce lr by 0.1 every ... epochs
scheduler = optax.piecewise_constant_schedule(init_value=0.002, boundaries_and_scales=[(2000, 0.5)])
# scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
# optimizer = optax.SGD(net.parameters(), lr=0.002, momentum=0.75)
optimizer = optax.chain(optax.sgd(learning_rate=0.002, momentum=0.75), scheduler)


for epoch in range(2):
    running_loss = 0.0
    # enumerate() is a function to keep track of the number of iterations in a loop
    for i, data in enumerate(train_ds, 0):
        # getting inputs, data is a list of [input, labels]
        input, labels = data
        # zero parameter gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(input)
        # calculating sqared mean error/difference between output and input (values)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if i % 1000 == 999:
            for param_group in optimizer.param_groups:
                if (param_group['momentum'] * 1.001) <= 0.99:
                    param_group['momentum'] *= 1.001
                    print(
                        f'New momentum: {param_group["momentum"]}, learning rate: {param_group["lr"]}')

        # print stats
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    scheduler.step()  # reducing learn rate

print('Finished training')

PATH = './cifar_net.pth'
flax.save(net.state_dict(), PATH)

with open(PATH, 'wb') as f:
    f.write(flax.serialization.to_bytes(net))



dataiter = iter(test_ds)
images, labels = next(dataiter)

grid = np.concatenate(images, axis=1)
imshow(grid)
print('Ground Truth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = Net()

with open(PATH, 'rb') as f:
    net = flax.serialization.from_bytes(net, f.read())

net.load_state_dict(flax.load(PATH, weights_only=True))

output = net(images)

_, predicted = flax.max(output, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

correct = 0
total = 0

with flax.no_grad():
    for data in test_ds:
        images, labels = data
        output = net(images)
        _, predicted = flax.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(
    f'Accuracy of the network on the 10000 test images: {100 * correct // total}%')

# count predictions per class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with flax.no_grad():
    for data in test_ds:
        images, labels = data
        output = net(images)
        _, predictions = flax.max(output, 1)
        # collecting prediction for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


device = flax.device('cuda:0' if flax.cuda.is_available() else 'cpu')

print(device)

# net.to(device)
