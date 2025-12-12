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
import pickle
# from JaxPR_Interpreter import eval_jaxpr

from jax.tree_util import tree_leaves

# for intervall
import os,sys

# Add the folder containing runtime dependencies first
os.add_dll_directory(r"C:\msys64\ucrt64\bin") # needed for Windows
sys.path.append(os.path.dirname(__file__))

# for direct access to the compiled module in build folder
sys.path.append(r"D:\Work\jax-interval-arithmetic")
import jax_new_interpreter



transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize(size=(4,4))
    ])

torch.manual_seed(0)

batch_size = 128

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0)

# for testing, we increase the batch size to 2 times the original size to speed up the process, as we dont learn anything from the test set
testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size*2, shuffle=False, num_workers=0)

classes = tuple(str(i) for i in range (10))

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


# @jax.jit
def relu(x):
    return jnp.maximum(x, 0)   # PReLU Intervall activation function
'''
# Convert elementwise, a float (c++ double) to an interval
def to_interval_matrix(arr):
    return ffi_module.IntervalMatrix([[ffi_module.toIntervalVal(float(x)) for x in row] for row in arr])

# Convert elementwise, a float (c++ double) to an interval, with range
def to_interval_matrix_range(arr, range):
    return ffi_module.IntervalMatrix([[ffi_module.toIntervalValRange(float(x), range) for x in row] for row in arr])

def sigmoid(x):
    return ffi_module.sigmoid(x)

'''


# @jax.jit
def init_params(key):
    # jax random split key into 5 keys, for each layer
    fc1_key, fc2_key, fc3_key, fc1_bias_key, fc2_bias_key, fc3_bias_key = jax.random.split(key, 6)
    
    params = {
    "fc1_kernel": jax.nn.initializers.glorot_uniform()(fc1_key,(16, 8),jnp.float32),  # (in_features, out_features)")
              "fc2_kernel": jax.nn.initializers.glorot_uniform()(fc2_key,(8, 4),jnp.float32),
              "fc3_kernel": jax.nn.initializers.glorot_uniform()(fc3_key,(4, 10),jnp.float32),
              # normalverteilung lequ normal
              "fc1_bias": jax.nn.initializers.lecun_normal()(fc1_bias_key,(1, 8), jnp.float32),
              "fc2_bias": jax.nn.initializers.lecun_normal()(fc2_bias_key,(1, 4), jnp.float32),
              "fc3_bias": jax.nn.initializers.lecun_normal()(fc3_bias_key,(1, 10), jnp.float32)}    # (kernel_height, kernel_width, in_channels, out_channels), HWIO
    return params

# @jax.jit
def Net(params, x):
    # Define a convolutional kernel manually
    
    # x = x.reshape(batch_size, 400)
    # x = x.reshape(x.shape[0], -1) # the size is (batch_size, 16*5*5) = (batch_size, 400), flattening the input to a vector of size 400
    # print("shape", x.shape)
    
    # take  above line out of function
    
    fc_k1 = params["fc1_kernel"]  # (in_features, out_features)
    # print(f"fc_k1 shape: {fc_k1.shape}")
    bias1 = params["fc1_bias"]
    # print(f"bias1 shape: {bias1.shape}")
    # print(f"x shape: {x.shape}")
    x = jnp.dot(x, fc_k1) + bias1  # (x.shape[0], x.shape[1]) @ (x.shape[1], 120) = (x.shape[0], 120)
    
    x = relu(x)
    
    fc_k2 = params["fc2_kernel"]  # (in_features, out_features)
    bias2 = params["fc2_bias"]
    x = jnp.dot(x, fc_k2) + bias2
    
    x = relu(x)
    
    fc_k4 = params["fc3_kernel"]  # (in_features, out_features)
    bias4 = params["fc3_bias"]
    x = jnp.dot(x, fc_k4) + bias4

    return x


# creating network object

# Initialize parameters
params = init_params(jax.random.PRNGKey(0))

OPEN = './cifar_net.pcl'

# with open(OPEN, 'rb') as file:
#     params = pickle.load(file)


# SGD
# removed optax.chain() as it is unnecessary for only a single optimizer
# optimizer = optax.sgd(learning_rate=0.01, momentum=0.9)

# Adam
# optimizer = optax.chain(optax.adam(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-08), optax.scale_by_adam(), optax.scale(-1.0)) # SGD(learning_rate=0.001, momentum=0.9))

# AdamW
# optimizer = optax.chain(optax.adamw(learning_rate=0.001, weight_decay=0.01), optax.scale_by_adam(), optax.scale(-1.0)) # SGD(learning_rate=0.001, momentum=0.9))

# optimizerState = optimizer.init(params)



# for modulo operation
print_loss = len(trainloader)//4
print_loss_ = len(trainloader)//4 - 1

dataiter = iter(testloader)
images, labels = next(dataiter)

images = images.numpy()
labels = labels.numpy()
images = jnp.moveaxis(images, 1, -1)

# output = Net(params, images)

# print(images.shape)
images = images.reshape(images.shape[0], -1)
# print(images.shape)



netJaxPR = jax.make_jaxpr(Net)(params, images)

# print("JAXPR", netJaxPR)
params_unfold = jax.tree.flatten(params)[0]

# print(len(params_unfold))


interval_Output_JaxPR = jax_new_interpreter.eval_jaxpr(netJaxPR.jaxpr, netJaxPR.literals, *params_unfold, images)
print(interval_Output_JaxPR)

# print(interval_Output_JaxPR)



'''
inputs = tree_leaves((params, images))
test = eval_jaxpr(netJaxPR.jaxpr, netJaxPR.literals, *inputs)
# test = eval_jaxpr(netJaxPR.jaxpr, netJaxPR.literals, images)
print(f"{test=}")


print('Ground Truth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

'''

'''
# print("params", params, type(params), "\nimages", images, type(images))
images_ = to_interval_matrix(images)
# print("\n images", images, type(images))
output = Net(params, images_)
# print("Output:" , output)

images_ = to_interval_matrix_range(images, 0.1)
'''
# print("\n images", images, type(images))
# output2 = Net(params, images_)
# print("Output:" , output2)

# ffi_module.checkValid(output,output2)

# net_jaxpr = jax.make_jaxpr(Net)(params, images)

# print(net_jaxpr)
'''
# uncomment and change output variable to test the jaxpr prediction
predicted = jnp.argmax(output, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

'''
correct = 0
total = 0
# count predictions per class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

'''
# @jax.jit
def predict(params, input):
    output_pred = Net(params, input)
    return output_pred

for data in tqdm(testloader):
    images, labels = data
    images = images.numpy()
    labels = labels.numpy()
    images = jnp.moveaxis(images, 1, -1)
    images = images.reshape(images.shape[0], -1)  # flattening the input to a vector of size 3072 (32*32*3)
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
        
PATH = './cifar_net.pcl'
'''
# with open(PATH, 'wb') as file:
#     pickle.dump(params, file)