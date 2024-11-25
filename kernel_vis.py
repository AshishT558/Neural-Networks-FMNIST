import cv2
import numpy
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
import matplotlib.pyplot as plt

conv_net = Conv_Net()
# conv_net.load_state_dict(torch.load('cnn.pth'))

# Get the weights of the first convolutional layer of the network

''' YOUR CODE HERE '''

c1_weights = conv_net.conv1.weight.data

# Create a plot that is a grid of images, where each image is one kernel from the conv layer.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, 
# the grid might have 4 rows and 8 columns. Finally, normalize the values in the grid to be 
# between 0 and 1 before plotting.

''' YOUR CODE HERE '''
def normalize_kernels(kernels):
    min_val = kernels.min()
    max_val = kernels.max()
    return (kernels - min_val) / (max_val - min_val)

normalized_kernels = normalize_kernels(c1_weights)

kernel_grid = torchvision.utils.make_grid(normalized_kernels, nrow=8, normalize=False, pad_value=1)

plt.figure(figsize=(12, 12))  
plt.imshow(kernel_grid.permute(1, 2, 0))
plt.axis('off')
plt.title("Kernels of the First Convolutional Layer")


# Save the grid to a file named 'kernel_grid.png'. Add the saved image to the PDF report you submit.
plt.savefig('kernel_grid.png', bbox_inches='tight')
plt.show()


# Apply the kernel to the provided sample image.

img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0					# Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)

print(img.shape)

# Apply the kernel to the image

output = F.conv2d(img, c1_weights, bias=None, stride=1, padding=1)


# convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.
# If not needed for your implementation, you can remove these lines.

output = output.squeeze(0)
output = output.unsqueeze(1)


# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, the grid might have 4 rows and 8 columns.
# Finally, normalize the values in the grid to be between 0 and 1 before plotting.

''' YOUR CODE HERE '''
output_images = []
for i in range(c1_weights.shape[0]):  
    single_kernel = c1_weights[i:i+1, :, :, :] 
    output = F.conv2d(img, single_kernel, bias=None, stride=1, padding=1)
    output_images.append(output)

output_images = torch.stack(output_images, dim=0)
normalized_outputs = normalize_kernels(output_images)
output_grid = torchvision.utils.make_grid(normalized_outputs, nrow=8, normalize=False, pad_value=1)

plt.figure(figsize=(12, 12))
plt.imshow(output_grid.permute(1, 2, 0))  # Permute to HWC format and move to CPU
plt.axis('off')
plt.title("Image Transformed by Kernels")

# Save the grid to a file named 'image_transform_grid.png'. Add the saved image to the PDF report you submit.
plt.savefig('image_transform_grid.png', bbox_inches='tight')
plt.show()

