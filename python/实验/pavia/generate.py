from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from Paper.degan_pavia.model.Generator import Generator
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=200, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--outf', default='dcgan/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--dataset', default='CIFAR', help='which dataset to train on, CIFAR|MNIST')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

###########   Load netG   ###########
#assert opt.netG != '', "netG must be provided!"
nc = 1

#1,5,40,50,100,500，550，600，700，750
net_p = torch.load('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_pavia/dcgan/netG_pavia40-750.pth')
netG = Generator(nc, opt.ngf, opt.nz)
netG.load_state_dict(net_p)

###########   Generate   ###########
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
noise = Variable(noise)

if(opt.cuda):
    netG.cuda()
    noise = noise.cuda()

noise.data.normal_(0,1)
fake = netG(noise)
data = fake.data
data = data.numpy()
np.save('feature200_no40-750',data)
vutils.save_image(fake.data,
            '%s/samples.png' % (opt.outf),
            normalize=True)
