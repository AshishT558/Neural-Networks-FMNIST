import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
from ffn import *

'''

In this file you will write end-to-end code to train two neural networks to categorize fashion-mnist data,
one with a feedforward architecture and the other with a convolutional architecture. You will also write code to
evaluate the models and generate plots.

'''


'''

PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted. Please do not change the transforms given below - the autograder assumes these.

'''

transform = transforms.Compose([                            # Use transforms to convert images to tensors and normalize them
    transforms.ToTensor(),                                  # convert images to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])             # Common method for grayscale images
])

batch_size = 500


'''

PART 2:
Load the dataset. Make sure to utilize the transform and batch_size from the last section.

'''

trainset = torchvision.datasets.FashionMNIST(download=True, transform=transform, train=True, root="./data")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(download=True, transform=transform, train=False, root="./data")
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


'''

PART 3:
Complete the model defintion classes in ffn.py and cnn.py. We instantiate the models below.

'''


feedforward_net = FF_Net()
conv_net = Conv_Net()



'''

PART 4:
Choose a good loss function and optimizer - you can use the same loss for both networks.

'''

criterion = nn.CrossEntropyLoss()

optimizer_ffn = optim.SGD(feedforward_net.parameters(), lr=0.01, momentum=0.9)
optimizer_cnn = optim.SGD(conv_net.parameters(), lr=0.01, momentum=0.9)

'''

PART 5:
Train both your models, one at a time! (You can train them simultaneously if you have a powerful enough computer,
and are using the same number of epochs, but it is not recommended for this assignment.)

'''


num_epochs_ffn = 30
loss_by_epoch_ffn = []
for epoch in range(num_epochs_ffn):  # loop over the dataset multiple times
    running_loss_ffn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Flatten inputs for ffn

        inputs = inputs.reshape(inputs.size(0), 28 * 28)

        # zero the parameter gradients
        optimizer_ffn.zero_grad()

        # forward + backward + optimize
        outputs = feedforward_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ffn.step()
        running_loss_ffn += loss.item()

    print(f"Training loss: {running_loss_ffn}")
    loss_by_epoch_ffn.append(running_loss_ffn)

print('Finished Training')

torch.save(feedforward_net.state_dict(), 'ffn.pth')  # Saves model file (upload with submission)


num_epochs_cnn = 30
loss_by_epoch_cnn = []
for epoch in range(num_epochs_cnn):  # loop over the dataset multiple times
    running_loss_cnn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer_cnn.zero_grad()

        # forward + backward + optimize
        outputs = conv_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()

    print(f"Training loss: {running_loss_cnn}")
    loss_by_epoch_cnn.append(running_loss_cnn)
print('Finished Training')

torch.save(conv_net.state_dict(), 'cnn.pth')  # Saves model file (upload with submission)


'''

PART 6:
Evalute your models! Accuracy should be greater or equal to 80% for both models.

Code to load saved weights commented out below - may be useful for debugging.

'''

# feedforward_net.load_state_dict(torch.load('ffn.pth'))
# conv_net.load_state_dict(torch.load('cnn.pth'))

correct_ffn = 0
total_ffn = 0

correct_cnn = 0
total_cnn = 0

fnn_found_correct = False
fnn_found_incorrect = False
cnn_found_correct = False
cnn_found_incorrect = False

fnn_classification = {}
cnn_classification = {}

with torch.no_grad():           # since we're not training, we don't need to calculate the gradients for our outputs
    for data in testloader:

        inputs, labels = data

        # FNN
        fnn_inputs = inputs.reshape(inputs.size(0), 28 * 28)
        outputs = feedforward_net(fnn_inputs)
        max_score, fnn_predicted = torch.max(outputs, 1)

        total_ffn += labels.size(0)
        for i in range(len(labels)):
            if not fnn_found_correct and fnn_predicted[i] == labels[i]:
                fnn_classification['correct'] = (inputs[i], labels[i], fnn_predicted[i])
                fnn_found_correct = True
            elif not fnn_found_incorrect and fnn_predicted[i] != labels[i]:
                fnn_classification['incorrect'] = (inputs[i], labels[i], fnn_predicted[i])
                fnn_found_incorrect = True
        
        correct_ffn += (fnn_predicted == labels).sum().item()
        

        # CNN
        outputs = conv_net(inputs)
        max_score, cnn_predicted = torch.max(outputs, 1)

        total_cnn += labels.size(0)
        for i in range(len(labels)):
            if not cnn_found_correct and cnn_predicted[i] == labels[i]:
                cnn_classification['correct'] = (inputs[i], labels[i], cnn_predicted[i])
                cnn_found_correct = True
            elif not cnn_found_incorrect and cnn_predicted[i] != labels[i]:
                cnn_classification['incorrect'] = (inputs[i], labels[i], cnn_predicted[i])
                cnn_found_incorrect = True

        correct_cnn += (cnn_predicted == labels).sum().item()

print('Accuracy for feedforward network: ', correct_ffn/total_ffn)
print('Accuracy for convolutional network: ', correct_cnn/total_cnn)


'''

PART 7:

Check the instructions PDF. You need to generate some plots. 

'''




'''
Plot of Loss by Epoch
'''
import matplotlib.pyplot as plt

# Create a figure with two subplots side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Plot FFN Loss on the first subplot
ax[0].plot(range(num_epochs_ffn), loss_by_epoch_ffn, label='FFN Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('FFN Loss Over Time')
ax[0].legend()

# Plot CNN Loss on the second subplot
ax[1].plot(range(num_epochs_cnn), loss_by_epoch_cnn, label='CNN Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].set_title('CNN Loss Over Time')
ax[1].legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
'''
Correctly Classified + Incorrectly Classified from each
'''
# Helper function to display an image with predictions
def display_image(image, true_label, predicted_label, title, subplot):
    image = image.squeeze()  # Remove channel dimension for grayscale images
    plt.subplot(subplot)
    plt.imshow(image, cmap='gray')
    plt.title(f"{title}\nTrue: {true_label}, Predicted: {predicted_label}")
    plt.axis('off')

# Plot results
plt.figure(figsize=(10, 8))

# FNN correct
display_image(
    fnn_classification['correct'][0],
    fnn_classification['correct'][1].item(),
    fnn_classification['correct'][2].item(),
    "FNN Correctly Classified",
    221
)

# FNN incorrect
display_image(
    fnn_classification['incorrect'][0],
    fnn_classification['incorrect'][1].item(),
    fnn_classification['incorrect'][2].item(),
    "FNN Incorrectly Classified",
    222
)

# CNN correct
display_image(
    cnn_classification['correct'][0],
    cnn_classification['correct'][1].item(),
    cnn_classification['correct'][2].item(),
    "CNN Correctly Classified",
    223
)

# CNN incorrect
display_image(
    cnn_classification['incorrect'][0],
    cnn_classification['incorrect'][1].item(),
    cnn_classification['incorrect'][2].item(),
    "CNN Incorrectly Classified",
    224
)

plt.tight_layout()
plt.show()

