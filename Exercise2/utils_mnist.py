import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from optparse import OptionParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_mnist(args):
    # MNIST dataset (images and labels)
    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args['batch_size'],
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args['batch_size'],
                                              shuffle=False)
    return train_loader, test_loader


def args_train():
    # Training Parameters
    parser = OptionParser()
    # Name of the Project
    parser.add_option('--model', dest='model', default='logistic_regression', type=str, help='type of the model')
    parser.add_option('--mode', type=str, default='dummy')
    parser.add_option('--port', type=str, default='dummy')
    parser.add_option('-f', type=str, default='dummy')
    (options, args) = parser.parse_args()
    return options


# define NN architecture
class MLP(nn.Module):
    def __init__(self, hidden_1, hidden_2, dropout=0):
        super(MLP, self).__init__()
        # number of hidden nodes in each layer (512)
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(64 * 64, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        self.droput = nn.Dropout(dropout)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 64 * 64)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add output layer
        x = self.fc3(x)
        return x


class MedicalMnist(data.Dataset):
    """
    Dataloader for the "medical mnist" dataset
    """
    def __init__(self, root, index):
        super(MedicalMnist, self).__init__()
        # scan all the subfolders under root using "glob"
        categories = sorted(glob.glob(root + '*'))
        # get all the subfolder names
        categories = [x.split('/')[-1] for x in categories]
        # scan all the image names under each subfolder
        imgs = dict()
        for c in categories:
            imgs[c] = sorted(glob.glob(os.path.join(root, c + '/*')))

        # load all the images and append them into a long list
        # also add labels, 0 for the first subfolder, 1, for the second subfolder....etc....
        all_labels = []
        all_imgs = []
        for i in range(len(categories)):
            c = categories[i]
            try:
                all_labels.append(i * np.ones(len(imgs[c]))[index])
                all_imgs.append(np.array([np.array(Image.open(y)) for y in imgs[c]])[index])
            except:
                index2 = range(index[0], len(imgs[c]))

                all_labels.append(i * np.ones(len(imgs[c]))[index2])
                all_imgs.append(np.array([np.array(Image.open(y)) for y in imgs[c]])[index2])

        # concatenate all the labels and images
        all_labels = np.concatenate(all_labels, 0)
        all_imgs = np.concatenate(all_imgs, 0)
        print('length of all images: ' + str(len(all_imgs)))

        # make the images and labels the attribute of the dataset
        self.categories = categories
        self.all_labels = all_labels
        self.all_imgs = all_imgs

    def __len__(self):
        # the dataloader need to know the length of the dataset
        return len(self.all_labels)

    def __getitem__(self, index):
        # normalize the image from 0-255 (because its 8-bit) to 0-1
        imgs = self.all_imgs[index] / 255
        labels = self.all_labels[index]
        # make the right data type: image need np.float32, label need np.uint8
        imgs = imgs.astype(np.float32)
        labels = labels.astype(np.uint8)
        return imgs, labels


def show_examples(images, labels):
    plt.figure(figsize=(20, 4))
    for index, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(1, len(images), index + 1)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title('Training:  ' + str(label))
    plt.show()


def get_medical_mnist(args):
    root = 'mmnist/'
    print('data folder:  ' + root)
    train_dataset = MedicalMnist(root=root, index=range(7000))
    validation_dataset = MedicalMnist(root=root, index=range(7000, 10000))

    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args['batch_size'],
                                               shuffle=True)

    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                              batch_size=args['batch_size'],
                                              shuffle=False)
    return train_loader, validation_loader


if __name__ == '__main__':
    root = 'Exercise2/mmnist/'
    train_dataset = MedicalMnist(root=root, index=range(7000))
    validation_dataset = MedicalMnist(root=root, index=range(7000, 10000))

    some_index = np.random.randint(0, len(train_dataset), 10)
    some_imgs = [train_dataset.__getitem__(idx)[0] for idx in some_index]
    some_labels = [train_dataset.__getitem__(idx)[1] for idx in some_index]

    show_examples(some_imgs[:7], some_labels[:7])






