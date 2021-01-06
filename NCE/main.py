import sys
sys.path.append(r'D:\github\Contrastive-learning\NCE')
import torch.optim as optim
from torch.utils.data import DataLoader as DataLoader
from torchvision import transforms
from NCE import *
from test import kNN
batch_size = 32
import Resnet as Resnet
import torch
import random
from NCEAverage import NCEAverage
from MNISTInstance import MNISTInstance
learning_rate = 10 ** random.uniform(-5, -4)   # From 1e-5 to 1e-4

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
criterion = NCECriterion(ndata)
x = 0
lemniscate = NCEAverage(inputSize= 128, outputSize = ndata, K = 1, T = 0.07, momentum = 0.5)
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
        loss = criterion(outputs)

        #print(loss.requires_grad)

        loss.backward()
        optimier.step()
        running_loss += loss.item()
        epoch_i =epoch_i+1
        x += 1
        if i %100 ==99:
            print('[%d,%5d]loss:%.3f'%(epoch+1,i+1,running_loss/100))
            running_loss = 0.0


optimier = optim.Adam(model.parameters(),lr=learning_rate)
if __name__ =='__main__':
    epoch_i = 0
    for epoch in range(200):
        train(epoch,model,train_loader,optimier,criterion)
        kNN(epoch, model, lemniscate, train_loader, test_loader, 200, 0.07)

