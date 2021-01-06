import torch
import numpy as np
class aliasmethod(object):
    def __init__(self,probs):
        if probs.sum()>1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)
        smaller = []
        larger = []
        for kk,prob in enumerate(probs):
            self.prob[kk] = probs[kk] * K
            if self.prob[kk] < 1:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) >0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = self.prob[large] - (1 - self.prob[small])
            if self.prob[large] < 1:
                smaller.append(large)
            else:
                larger.append(large)
        for last_one in smaller+larger:
            self.prob[last_one] = 1
    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()
    def sample(self,N):
        '''


        :param N: draw N sample from multinomial
        :return:
        '''
        K = self.alias.size(0)
        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0,kk)
        alias = self.alias.index_select(0,kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())
        return oq+oj
