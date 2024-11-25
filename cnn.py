import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a convolutional neural network. 

Please only complete the model definition and do not include any training code.

The model should be a convolutional neural network, that accepts 28x28 grayscale images as input, and outputs a tensor of size 10.
The number of layers/kernels, kernel sizes and strides are up to you. 

Please refer to the following for more information about convolutions, pooling, and convolutional layers in PyTorch:

    - https://deeplizard.com/learn/video/YRhxdVk_sIs
    - https://deeplizard.com/resource/pavq7noze2
    - https://deeplizard.com/resource/pavq7noze3
    - https://setosa.io/ev/image-kernels/
    - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html


Whether you need to normalize outputs using softmax depends on your choice of loss function. PyTorch documentation is available at
https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton requires normalized outputs or not.

'''

class Conv_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3)

         # Linear layers
        self.fc1 = nn.Linear(64 * 6 * 6, 600)
        self.fc2 = nn.Linear(600, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 50)
        self.fc5 = nn.Linear(50, 10)

    def forward(self, x):

        # convolutions + pooling
        c1 = F.relu(self.conv1(x))

        s2 = F.max_pool2d(c1, kernel_size=2, stride=2)

        c3 = F.relu(self.conv2(s2))

        s4 = F.max_pool2d(c3, 2)

        # Flatten 
        s4 = torch.flatten(s4, 1)

        # fully connected layers
        f5 = F.relu(self.fc1(s4))
        
        f6 = F.relu(self.fc2(f5))

        f7 = F.relu(self.fc3(f6))

        f8 = F.relu(self.fc4(f7))

        output = self.fc5(f8)

        return output
        
