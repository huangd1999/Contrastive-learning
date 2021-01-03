import torch
import sys
sys.path.append(r'D:\github\Contrastive-learning\NCE')
import os
import torchvision
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import DataLoader as DataLoader
from torchvision import datasets
from torchvision import transforms
from NCE import *
import numpy as np

batch_size = 2
from PIL import Image
import torchvision.datasets as datasets
import Resnet as Resnet
import torch
from torch.autograd import Function
from torch import nn
from aliasmethod import aliasmethod
import math


class NCEFunction(Function):
    @staticmethod
    def forward(self, x, y, memory, idx, params):
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()

        momentum = params[3].item()
        outputSize = memory.size(0)
        inputSize = memory.size(1)
        batchSize = x.size(0)

        # sample positives & negatives
        idx.select(1, 0).copy_(y.data)

        # sample correspoinding weights
        weight = torch.index_select(memory, 0, idx.view(-1))
        weight.resize_(batchSize, K + 1, inputSize)

        # inner product
        # out = torch.mm(x.data, memory.t())
        # out.div_(T) # batchSize * N
        x_change = x.reshape(batchSize, inputSize, 1)
        out = torch.bmm(weight, x_change)
        out.div_(T).exp_()  # batchSize * self.K+1
        x.data.resize_(batchSize, inputSize)
        if Z < 0:
            params[2] = out.mean() * outputSize
            Z = params[2].item()
            print("normalization constant Z is set to {:.1f}".format(Z))
        out.div_(Z).resize_(batchSize, K + 1)

        self.save_for_backward(x, memory, y, weight, out, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, weight, out, params = self.saved_tensors
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()
        batchSize = gradOutput.size(0)
        # gradients d Pm / d linear = exp(linear) / Z
        gradOutput.data.mul_(out.data)
        # add temperature
        gradOutput.data.div_(T)
        gradOutput.resize_(batchSize, 1, K + 1)
        # gradient of linear
        gradInput = torch.bmm(gradOutput, weight)
        gradInput.resize_as_(x)
        # update the non-parametric data
        # weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos = weight.select(1, 0).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return gradInput, None, None, None, None


class NCEaverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.9, Z=None):
        super(NCEaverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = aliasmethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]));
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        batchSize = x.size(0)
        idx = self.multinomial.sample(batchSize * (self.K + 1)).view(batchSize, -1)
        out = NCEFunction.apply(x, y, self.memory, idx, self.params)
        return out
class MNISTInstance(datasets.MNIST):
    """MNIST Instance Dataset.
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=28, scale=(0.2,1.)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),
])

trainset = MNISTInstance(root='../data/mnist', train=True, download=True, transform=transform_train)

testset = MNISTInstance(root='../data/mnist', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

cuda = True if torch.cuda.is_available() else False
ndata = trainset.__len__()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Resnet.ResNet50().cuda()
x_list = []
y_list = []
criterion = NCECriterion(ndata)
x = 0
lemniscate = NCEaverage(inputSize= 128, outputSize = ndata, K = 1, T = 0.07, momentum = 0.5)
lemniscate.to(device)
criterion.to(device)
def train(epoch,model,train_loader,optimier,criterion):
    global x
    running_loss = 0.0
    epoch_i = 0
    for i,(inputs, _, index) in enumerate(train_loader, 0):
        inputs = inputs.cuda()
        index = index.cuda()
        optimier.zero_grad()
        features = model(inputs)
        outputs = lemniscate(features, index)
        loss = criterion(outputs, index)

        #print(loss.requires_grad)

        loss.backward()
        optimier.step()
        running_loss += loss.item()
        epoch_i =epoch_i+1
        x_list.append(x)
        x += 1
        y_list.append(loss.item())
        if i %2 ==1:
            print('[%d,%5d]loss:%.3f'%(epoch+1,i+1,running_loss/300))
            running_loss = 0.0
optimier = optim.Adam(model.parameters(),lr=0.00001)
if __name__ =='__main__':
    epoch_i = 0
    for epoch in range(10):
        train(epoch,model,train_loader,optimier,criterion)
    plt.plot(x_list, y_list)
    plt.xlabel('epoch_train')
    plt.ylabel('loss_train')
    plt.show()


