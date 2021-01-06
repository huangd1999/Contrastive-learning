import os
import sys
sys.path.append(r'D:\github\Contrastive-learning\NCE')
import torch.optim as optim
from torch.utils.data import DataLoader as DataLoader
from torchvision import transforms
from NCE import *
from test import kNN
from AverageMeter import *
import time
import Resnet as Resnet
import torch
from NCEAverage import NCEAverage
from MNISTInstance import CIFAR10Instance
import argparse
import torch.backends.cudnn as cudnn

batch_size = 128
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = CIFAR10Instance(root='../data/cifar10', train=True, download=True, transform=transform_train)

testset = CIFAR10Instance(root='../data/cifar10', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
ndata = trainset.__len__()

print('==> Building model..')
net = Resnet.ResNet18().cuda()
# define leminiscate

lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)


if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# Model
if args.test_only or len(args.resume) > 0:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + args.resume)
    net.load_state_dict(checkpoint['net'])
    lemniscate = checkpoint['lemniscate']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# define loss function
if hasattr(lemniscate, 'K'):
    criterion = NCECriterion(ndata)
else:
    criterion = nn.CrossEntropyLoss()

net.to(device)
lemniscate.to(device)
criterion.to(device)

if args.test_only:
    acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
    sys.exit(0)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if epoch >= 80:
        lr = args.lr * (0.1 ** ((epoch - 80) // 40))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
        optimizer.zero_grad()

        features = net(inputs)
        outputs = lemniscate(features, indexes)
        loss = criterion(outputs)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
            epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    acc = kNN(epoch, net, lemniscate, trainloader, testloader, 200, args.nce_t, 0)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'lemniscate': lemniscate,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

    print('best accuracy: {:.2f}'.format(best_acc * 100))

acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
print('last accuracy: {:.2f}'.format(acc * 100))