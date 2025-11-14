import flax
import flax.linen
from flax import linen as lnn
import optax
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
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

'''
# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
'''

# get some random training images from the dataloader, as trainloader is an iterable object, that is set to shuffle
dataiter = iter(trainloader)
images, labels = next(dataiter)


@jax.jit
def prelu(x, alpha=0.1):
    return jnp.where(x > 0, x, alpha * x)  # PReLU activation function

def max_pool(x, window_shape=(2,2), strides=(2,2), padding='VALID'):
    return lax.reduce_window(
        operand=x, # input tensor
        init_value=-jnp.inf, # initial value for the reduction
        computation=lax.max, # reduction operation
        window_dimensions=(1, *window_shape, 1 ), # window shape, (1, 2, 2, 1) means that we want to reduce the height and width of the input tensor by a factor of 2
        window_strides=(1, *strides, 1), # strides, (1, 2, 2, 1) means that we want to move the window by 2 pixels in height and width
        padding=padding  
    )
max_pool = jax.jit(max_pool, static_argnames=('window_shape', 'strides', 'padding'))

@jax.jit
def init_params(key):
    # jax random split key into 5 keys, for each layer
    conv1_key, conv2_key, fc1_key, fc2_key, fc3_key, fc1_bias_key, fc2_bias_key, fc3_bias_key = jax.random.split(key, 8)
    
    params = {"conv1_kernel": jax.nn.initializers.glorot_uniform(2, 3)(conv1_key,(5, 5, 3, 6), jnp.float32), # (kernel_height, kernel_width, in_channels, out_channels), HWIO)
              "conv1_bias": jnp.zeros(6,), 
              "conv2_kernel": jax.nn.initializers.glorot_uniform(2, 3)(conv2_key,(5, 5, 6, 16), jnp.float32), 
              "conv2_bias": jnp.zeros(16,),
              "fc1_kernel": jax.nn.initializers.glorot_uniform()(fc1_key,(400, 120),jnp.float32),  # (in_features, out_features)")
              "fc2_kernel": jax.nn.initializers.glorot_uniform()(fc2_key,(120, 84),jnp.float32),
              "fc3_kernel": jax.nn.initializers.glorot_uniform()(fc3_key,(84, 10),jnp.float32),
              # normalverteilung lequ normal
              "fc1_bias": jax.nn.initializers.lecun_normal()(fc1_bias_key,(120, 1), jnp.float32).squeeze(),
              "fc2_bias": jax.nn.initializers.lecun_normal()(fc2_bias_key,(84, 1), jnp.float32).squeeze(),
              "fc3_bias": jax.nn.initializers.lecun_normal()(fc3_bias_key,(10, 1), jnp.float32).squeeze()

}    # (kernel_height, kernel_width, in_channels, out_channels), HWIO
    return params

@jax.jit
def Net(params, x):
    # Define a convolutional kernel manually
    x = lax.conv_general_dilated(
        lhs=x, # lhs is the input
        rhs=params["conv1_kernel"], # rhs is the kernel
        window_strides=(1, 1), # strides, this is the amount of pixels to move the kernel at each step
        padding='VALID', # padding, this is the amount of pixels to add to the input, VALID means no padding, SAME means padding so that the output has the same size as the input
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')#, # ('input_format', 'kernel_format', 'output_format'),
            # Input (NHWC): The input tensor has dimensions ordered as: N: Batch size. H: Height. W: Width. C: Channels.
            # Kernel (HWIO): The kernel tensor has dimensions ordered as: H: Height of the filter. W: Width of the filter. I: Input channels. O: Output channels.
            # Output (NHWC): The output tensor has dimensions ordered as: N: Batch size. H: Height. W: Width. C: Channels.
            
        # precision=jax.lax.Precision.HIGHEST
    ) #+ params["conv1_bias"]
    
    x = prelu(x)  # PReLU activation function
    
    # fills outer space so that all numbers of input are captured
    x = max_pool(x) # , window_shape=(2, 2), strides=(2, 2), padding='VALID')
    
    x = lax.conv_general_dilated(
        lhs = x, # input = x
        rhs = params["conv2_kernel"],
        window_strides=(1, 1), # strides, moves one pixel at a time
        padding='VALID', # padding, no padding
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')#, # ('input_format', 'kernel_format', 'output_format'),
    ) + params["conv2_bias"]
    
    x = prelu(x)
    # fills outer space so that all numbers of input are captured
    x = max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
    
    # x = x.reshape(batch_size, 400)
    x = x.reshape(x.shape[0], -1) # the size is (batch_size, 16*5*5) = (batch_size, 400), flattening the input to a vector of size 400
    fc_k1 = params["fc1_kernel"]  # (in_features, out_features)
    bias1 = params["fc1_bias"]
    x = jnp.dot(x, fc_k1) + bias1  # (x.shape[0], x.shape[1]) @ (x.shape[1], 120) = (x.shape[0], 120)
    
    x = prelu(x)
    
    fc_k2 = params["fc2_kernel"]  # (in_features, out_features)
    bias2 = params["fc2_bias"]
    x = jnp.dot(x, fc_k2) + bias2
    
    x = prelu(x)
    
    fc_k3 = params["fc3_kernel"]  # (in_features, out_features)
    bias3 = params["fc3_bias"]
    x = jnp.dot(x, fc_k3) + bias3
    return x


# creating network object

# Initialize parameters
params = init_params(jax.random.PRNGKey(0))

# SGD
# removed optax.chain() as it is unnecessary for only a single optimizer
optimizer = optax.sgd(learning_rate=0.01, momentum=0.9)

# Adam
# optimizer = optax.chain(optax.adam(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-08), optax.scale_by_adam(), optax.scale(-1.0)) # SGD(learning_rate=0.001, momentum=0.9))

# AdamW
# optimizer = optax.chain(optax.adamw(learning_rate=0.001, weight_decay=0.01), optax.scale_by_adam(), optax.scale(-1.0)) # SGD(learning_rate=0.001, momentum=0.9))

optimizerState = optimizer.init(params)


@jax.jit
def loss(params, input, labels):  # corrected the loss function to take labels as input
    output = Net(params, input)
    # calculating sqared mean error/difference between output and input (values)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(output, labels))
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
for epoch in range(2):
    running_loss = 0.0
    # enumerate() is a function to keep track of the number of iterations in a loop
    for i, data in enumerate(tqdm(trainloader), 0):
        # getting inputs, data is a list of [input, labels]
        input, labels = data
        input = input.numpy()
        labels = labels.numpy()
        input = jnp.moveaxis(input, 1, -1)

        params, optimizerState, lossVal = train_step(
            params, optimizerState, input, labels)  # combined functions for better jit

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


output = Net(params, images)

predicted = jnp.argmax(output, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

correct = 0
total = 0
# count predictions per class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}


@jax.jit
def predict(params, input):
    output_pred = Net(params, input)
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