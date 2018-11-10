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
import scipy.io as sio
import torchvision.utils as vutils
from torch.autograd import Variable
from model.Discriminator import Discriminator
from model.Generator import Generator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.utils.data as Data
import sklearn.preprocessing as prep


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--imageSize', type=int, default=145, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='dcgan/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
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

###############   DATASET   ##################
#data = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Indian/Indian_pines.mat')['indian_pines']

label = sio.loadmat('/media/jiming/_E/angelo/Indian/Indian_pines_gt.mat')['indian_pines_gt']
data = sio.loadmat('/media/jiming/_E/angelo/Indian/Indian_pines_corrected.mat')['indian_pines_corrected']

#######提样本


#######

# label = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/ALofauto/PaviaU_gt.mat')['paviaU_gt']
# data = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/ALofauto/PaviaU.mat')['paviaU']

#data = np.reshape(data,[145*145,220],order='F')
#data_p = prep.scale(data)
# pca = PCA(n_components=3)
# # data = np.array(data)
# data = pca.fit_transform(data)
#data = np.reshape(data_p,[145,145,220],order='F')
# plt.imshow(data, cmap='gray')
# plt.show()
random.seed(700)        #1,5,40,50,100,500,13
index = [i for i in range(200)]
random.shuffle(index)


data_new = []
for i in range(30):
    pic = data[:,:,index[i]]
    data_new.append(pic)


train_y = np.ones(30)
train_y = torch.FloatTensor(train_y)
data_new = np.array(data_new,dtype='float')
data_new = torch.from_numpy(data_new)
data_new=torch.unsqueeze(data_new,dim=1)

print(data_new.size())
vutils.save_image(data_new,
            '/media/jiming/_E/angelo/Paper/degan_indian/dcgan/s_samples30.png',nrow=6,normalize=True)

print(data_new.size())
dataset = Data.TensorDataset(data_tensor=data_new, target_tensor=train_y)
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = opt.batchSize,
                                     shuffle = True)

###############   MODEL   ####################
ndf = opt.ndf
ngf = opt.ngf
nc = 1


netD = Discriminator(nc, ndf)
netG = Generator(nc, ngf, opt.nz)
#if(opt.cuda):
netD.cuda()
netG.cuda()

###########   LOSS & OPTIMIZER   ##########
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

##########   GLOBAL VARIABLES   ###########
#noise_all = torch.FloatTensor(20,opt.nz,1,1)
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
real = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0
#noise_all = Variable(noise_all)
noise = Variable(noise)
real = Variable(real)
label = Variable(label)
#if(opt.cuda):
    #noise_all = noise_all.cuda()
noise = noise.cuda()
real = real.cuda()
label = label.cuda()

########### Training   ###########
for epoch in range(1,opt.niter+1):
    for i, (images,_) in enumerate(loader):
        ########### fDx ###########
        netD.zero_grad()
        # train with real data, resize real because last batch may has less than
        # opt.batchSize images
        real.data.resize_(images.size()).copy_(images)
        label.data.resize_(images.size(0)).fill_(real_label)

        output = netD(real)
        errD_real = criterion(output, label)
        errD_real.backward()

        # train with fake data
        label.data.fill_(fake_label)
        noise.data.resize_(images.size(0), opt.nz, 1, 1)
        #noise = torch.FloatTensor(np.random.uniform(-0.1,0.1,noise.size())).cuda()
        #noise = torch.rand(-1,1,noise.size()).cuda()
        #noise = Variable(noise)
        #print(noise.size())
        noise.data.normal_(0,1).cuda()
        #fake_all = netG(noise_all)
        fake = netG(noise)
        # detach gradients here so that gradients of G won't be updated
        output = netD(fake.detach())
        errD_fake = criterion(output,label)
        errD_fake.backward()


        optimizerD.step()
        errD = errD_fake + errD_real
        ########### fGx ###########
        netG.zero_grad()
        label.data.fill_(real_label)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        ########### Logging #########
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f '
                  % (epoch, opt.niter, i, len(loader),
                     errD.data[0], errG.data[0]))

        ########## Visualize #########
        if(i % 50 == 0):
            outpic = fake.data
            #print(outpic.size())
            # outpic = outpic.numpy()
            # print(outpic.shape)
            #outpic = outpic[:4,:,:,:]
            vutils.save_image(outpic,
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
            # vutils.save_image(fake_all.data,
            #                   '%s/fake_all_epoch_%03d.png' % (opt.outf, epoch),
            #                   normalize=True)

torch.save(netG.state_dict(), '%s/netG_noindian30-epoch1' % (opt.outf))
#torch.save(netD.state_dict(), '%s/netD_noindian50R.pth' % (opt.outf))
