import flax
import flax.linen
from flax import linen as lnn
from flax import nnx
# from flax.training import train_state # can be used later
import optax
import jax
import jax.numpy as jnp
from jax import jit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)), # added Gaussian blur to the image, much slower, and some bit worse accuracy
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #slows down training by 2x, and slightly less accuracy
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

torch.manual_seed(0)

batch_size = 128

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0)

# for testing, we increase the batch size to 2 times the original size to speed up the process, as we dont learn anything from the test set
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size*2, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images from the dataloader, as trainloader is an iterable object, that is set to shuffle
dataiter = iter(trainloader)
images, labels = next(dataiter)


class Net(flax.linen.Module):

    @flax.linen.compact
    def __call__(self, x):
        x = lnn.Conv(features=6, kernel_size=(5, 5))(x)
        x = lnn.PReLU()(x)
        # fills outer space so that all numbers of input are captured
        x = lnn.max_pool(x, window_shape=(
            2, 2), strides=(2, 2), padding='VALID')
        x = lnn.Conv(features=16, kernel_size=(5, 5))(x)
        x = lnn.PReLU()(x)
        # fills outer space so that all numbers of input are captured
        x = lnn.max_pool(x, window_shape=(
            2, 2), strides=(2, 2), padding='VALID')
        x = x.reshape(x.shape[0], -1)
        x = lnn.Dense(features=120)(x)
        x = lnn.PReLU()(x)
        x = lnn.Dense(features=84)(x)
        x = lnn.PReLU()(x)
        x = lnn.Dense(features=10)(x)
        return x


# creating network object
net = Net()

# Initialize parameters
params = net.init(jax.random.PRNGKey(0), jnp.zeros(
    (1, 32, 32, 3)))  # init network ['params']

# SGD
# removed optax.chain() as it is unnecessary for only a single optimizer

total_steps = 4 * len(trainloader) # total number of steps in the training process
warmup_steps = int(0.1 * total_steps*2)


schedule = optax.join_schedules(
    schedules=[
        optax.linear_schedule(0.002, 0.05, warmup_steps),  # warmup
        optax.cosine_decay_schedule(0.05, total_steps - warmup_steps, 0.005) # decay, learning rate will be 0.001 at the end of training
    ],
    boundaries=[warmup_steps] # boundaries is the point where the schedule changes from one to another
)

optimizer = optax.sgd(learning_rate=schedule, momentum=0.9)

# optimizer = optax.sgd(learning_rate=0.01, momentum=0.9)

# Adam
# optimizer = optax.chain(optax.adam(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-08), optax.scale_by_adam(), optax.scale(-1.0)) # SGD(learning_rate=0.001, momentum=0.9))

# AdamW
# optimizer = optax.chain(optax.adamw(learning_rate=0.001, weight_decay=0.01), optax.scale_by_adam(), optax.scale(-1.0)) # SGD(learning_rate=0.001, momentum=0.9))

optimizerState = optimizer.init(params)


def loss(params, input, labels):  # corrected the loss function to take labels as input
    output = net.apply(params, input)
    # calculating sqared mean error/difference between output and input (values)
    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(output, labels))
    return loss


# jitting the loss function
# val_and_grad_jit = jax.jit(jax.value_and_grad(loss, argnums=0)) # added argnums=0 to specify that the first argument is the one we want to differentiate with respect to


@jax.jit
def train_step(params, optimizerState, input, labels):
    lossVal, gradient = jax.value_and_grad(loss)(params, input, labels)  # jit
    # update params (using optimizer)
    param_updates, optimizerState = optimizer.update(gradient, optimizerState)
    params = optax.apply_updates(params, param_updates)
    return params, optimizerState, lossVal


# for modulo operation
print_loss = len(trainloader)//4
print_loss_ = len(trainloader)//4 - 1

# training network
for epoch in range(4):
    running_loss = 0.0
    # enumerate() is a function to keep track of the number of iterations in a loop
    for i, data in enumerate(tqdm(trainloader), 0):
        # getting inputs, data is a list of [input, labels]
        input, labels = data
        input = input.numpy()
        labels = labels.numpy()
        input = jnp.moveaxis(input, 1, -1)

        params, optimizerState, lossVal = train_step(
            params, optimizerState, input, labels)  # combined funtions for better jit

        running_loss += lossVal
        if i % print_loss == print_loss_:
            # print stats
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_loss:.3f}')
            running_loss = 0.0

print('Finished training')

PATH = './cifar_net.pth'

dataiter = iter(testloader)
images, labels = next(dataiter)

images = images.numpy()
labels = labels.numpy()
images = jnp.moveaxis(images, 1, -1)

print('Ground Truth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


output = net.apply(params, images)

predicted = jnp.argmax(output, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

correct = 0
total = 0
# count predictions per class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}


@jax.jit
def predict(params, input):
    output_pred = net.apply(params, input)
    return output_pred


for data in tqdm(testloader):
    images, labels = data
    images = images.numpy()
    labels = labels.numpy()
    images = jnp.moveaxis(images, 1, -1)
    output = predict(params, images)
    predicted = jnp.argmax(output, 1)
    total += labels.shape[0]
    correct += int((predicted == labels).sum())
    
    for label, prediction in zip(labels, predicted):
        label = int(label)
        prediction = int(prediction)
        if label == prediction:
            correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1

accuracy = 100 * float(correct) / total

print(f'Overall test accuracy: {accuracy:.2f}%')

# per class accuracy
for classname in classes:
    if total_pred[classname] > 0:
        accuracy = 100 * float(correct_pred[classname]) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')