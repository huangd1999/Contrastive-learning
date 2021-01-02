import torch
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
from NCE import NCECriterion
import numpy as np
batch_size = 2
from PIL import Image
import torchvision.datasets as datasets

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
class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(x))
        return F.relu(x+y)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=3)
        self.conv2 = nn.Conv2d(10,100,kernel_size=3)
        self.conv3 = nn.Conv2d(100,1000,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(1000,10000,kernel_size=3,padding=1)

        self.mp = nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(10)
        self.rblock2 = ResidualBlock(100)
        self.rblock3 = ResidualBlock(1000)
        self.rblock4 = ResidualBlock(10000)

        self.fc = nn.Linear(10000,60000)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = self.mp(F.relu(self.conv3(x)))
        x = self.rblock3(x)
        x = self.mp(F.relu(self.conv4(x)))
        x = self.rblock4(x)

        x = x.view(in_size,-1)
        x = self.fc(x)
        return x
model =Net().cuda()
x_list = []
y_list = []
criterion = NCECriterion(ndata)
x = 0
def train(epoch,model,train_loader,optimier,criterion):
    global x
    running_loss = 0.0
    epoch_i = 0
    for i,(inputs, _, index) in enumerate(train_loader, 0):
        inputs = inputs.cuda()
        index = index.cuda()
        optimier.zero_grad()

        #forward,backward,updata
        outputs = model(inputs)
        print(outputs)
        loss = criterion(outputs,index)
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