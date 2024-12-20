import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a feedforward neural network. 

Please only complete the model definition and do not include any training code.

The model should be a feedforward neural network, that accepts 784 inputs (each image is 28x28, and is flattened for input to the network)
and the output size is 10. Whether you need to normalize outputs using softmax depends on your choice of loss function.

PyTorch documentation is available at https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton 
requires normalized outputs or not.

'''

class FF_Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Linear layers
        self.fc1 = nn.Linear(784, 600)
        self.fc2 = nn.Linear(600, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 50)
        self.fc5 = nn.Linear(50, 10)

    def forward(self, x):
        
        l1 = F.relu(self.fc1(x))
        l2 = F.relu(self.fc2(l1))
        l3 = F.relu(self.fc3(l2))
        l4 = F.relu(self.fc4(l3))
        output = self.fc5(l4)

        return output
        
