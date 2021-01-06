import torch
from torch.autograd import Function
from torch import nn
from lib.aliasmethod import aliasmethod
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


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.9, Z=None):
        super(NCEAverage, self).__init__()
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
