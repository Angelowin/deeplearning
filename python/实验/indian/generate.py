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
from model.Generator import Generator
import numpy as np

import scipy.stats as ss






parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
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
net_p = torch.load('/media/jiming/_E/angelo/Paper/degan_indian/dcgan/netG_noindian30-epoch1')
netG = Generator(nc, opt.ngf, opt.nz)
netG.load_state_dict(net_p)

###########   Generate   ###########
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
noise = Variable(noise)

if(opt.cuda):
    netG.cuda()
    noise = noise.cuda()

noise.data.normal_(0,1)
#noise = torch.rand(noise.size())
# noise = torch.FloatTensor(np.random.uniform(-0.1,0.1,noise.size()))
# noise = Variable(noise)

fake = netG(noise)
data = fake.data
data = data.numpy()
np.save('feature100_no-700-epoch1',data)     #1,5,40,50,100,500,550,600,700,750

vutils.save_image(fake.data,
           '/media/jiming/_E/angelo/Paper/degan_indian/dcgan/g_samples6_700.png',nrow=8,normalize=True)     #'%s/g_samples.png' % (opt.outf),

#import matplotlib.pyplot as plt

# fake = np.load('/media/jiming/_E/angelo/Paper/degan_indian/dcgan/netG_noindian5-160.pth')
# # data_new = torch.from_numpy(fake)
# # data_new=torch.squeeze(data_new,dim=1).numpy()
# #
# # data_new = data_new.transpose(1, 2, 0)
# #
# # from scipy import io
# # io.savemat('no50_5.mat',{'no50_5':data_new})
#
# vutils.save_image(fake,
#             '/media/jiming/_E/angelo/Paper/degan_indian/dcgan/g_samples_160.png',nrow=5,normalize=True)     #'%s/g_samples.png' % (opt.outf),

# fake=torch.squeeze(data_new,dim=1)
# for i in range(10):
#     plt.figure(i)
#     fake1 = fake[i,:,:]
#     fake1 = fake1.numpy()
#     fake1 = np.array(fake1,dtype=float)
#     plt.imshow(fake1,cmap='gray')
#     plt.show()