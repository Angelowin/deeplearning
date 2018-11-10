from sklearn import svm
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.io as sio
import numpy as np
from pylab import *
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import itertools
import torch.utils.data as Data
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import random
########################################modify#################################################################

data_p = '/media/jiming/_E/angelo/CGAN/Spectual.mat'
a = sio.loadmat(data_p)
x_spe = a['x_spe']
s_label = a['s_label']
data = torch.FloatTensor(x_spe)

def one_hot(label,num):
    o = [1,2,3,4,5,6,7,8,9]
    o = list(o)
    o = np.reshape(o,[9,1])
    one_hot = OneHotEncoder()
    one_hot.fit(o)
    label = np.reshape(label,[num,1],order='F')
    a = one_hot.transform(label).toarray()
    return a


label = one_hot(s_label,270)
# data = data[30:60]
# label = label[30:60]
label = torch.FloatTensor(label)

torch_dataset = Data.TensorDataset(data_tensor=data, target_tensor=label)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = 3,
    shuffle = True,
    num_workers=1
)


class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.fc1_1 = nn.Linear(100, 128)
        self.fc1_1_bn = nn.BatchNorm1d(128)
        self.fc1_2 = nn.Linear(9, 128)
        self.fc1_2_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(256, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 103)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.tanh(self.fc4(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1_1 = nn.Linear(103, 256)
        self.fc1_2 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.fc1_1(input), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = F.sigmoid(self.fc4(x))

        return x

#
netD = discriminator().cuda()
netG = generator().cuda()
print(netD)
print (netG)
#
# ###########   LOSS & OPTIMIZER   ##########
criterion = nn.MSELoss()
optimizerD = torch.optim.SGD(netD.parameters(),lr=0.0002)
optimizerG = torch.optim.SGD(netG.parameters(),lr=0.0002)

##########   GLOBAL VARIABLES   ###########
noise = torch.FloatTensor(3, 100, 1)
y_t = torch.FloatTensor(3)
y_t = Variable(y_t).cuda()
noise = Variable(noise).cuda()
real_label = 1
fake_label = 0


########### Training   ###########
for epoch in range(1,100000):
    for step, (batch_x, batch_y) in enumerate(loader):
        ########### fDx ###########
        netD.zero_grad()
        batch_y = Variable(batch_y).cuda()
        batch_x = Variable(batch_x).cuda()
        # x1 = torch.cat((batch_x,batch_y),1)

        output = netD(batch_x,batch_y)
        y_t.data.fill_(real_label)
        errD_real = criterion(output, y_t)
        errD_real.backward()

        # train with fake data
        y_t.data.fill_(fake_label)
        noise.data.resize_(3,100)
        noise.data.normal_(0,1)

        # noise_ = torch.cat((noise, batch_y), 1)
        fake = netG(noise,batch_y)
        # detach gradients here so that gradients of G won't be updated
        # fake_ = torch.cat((fake,batch_y),1)
        output = netD(fake.detach(),batch_y.detach())
        errD_fake = criterion(output,y_t)
        errD_fake.backward()

        errD = errD_fake + errD_real
        optimizerD.step()

        ########### fGx ###########
        netG.zero_grad()
        y_t.data.fill_(real_label)
        output = netD(fake,batch_y)
        errG = criterion(output, y_t)
        errG.backward()
        optimizerG.step()


    ########### Logging #########
    if(epoch%10==0):
        print("epoch:",epoch,"Loss_D:" , errD.data[0] , "Loss_G:" ,errG.data[0])
    if(epoch%2000==0):
        num = np.arange(103)
        plt.figure(epoch)
        plot(num,fake.data[0].cpu().numpy().T)
        plot(num,batch_x.data[0].cpu().numpy().T)
        savefig('/media/jiming/_E/angelo/CGAN/Pavia_centre/MyFig_{}.jpg'.format(epoch))


        outf = '/media/jiming/_E/angelo/CGAN/Pavia_centre'
        torch.save(netG.state_dict(), '%s/netG2_%s.pth' % (outf,epoch))
        torch.save(netD.state_dict(), '%s/netD2_%s.pth' % (outf,epoch))
outf = '/media/jiming/_E/angelo/CGAN/Pavia_centre'
torch.save(netG.state_dict(), '%s/netG11.pth' % (outf))
torch.save(netD.state_dict(), '%s/netD11.pth' % (outf))





######################################################
######################################################



netG.load_state_dict(torch.load('/media/jiming/_E/angelo/CGAN/Pavia_centre/netG.pth'))

netG.eval()

noise = torch.FloatTensor(64, 20, 1)
noise = Variable(noise).cuda()
fake_b = torch.ones(1,103)
label = torch.ones(1,9)
label__ = array(1)
for i in range(1,10):
    a = i
    print(a)
    b = np.tile(a,(1,500))
    label_fake = np.tile(a,500)
    label__ = np.hstack((label__,label_fake))
    c = one_hot(b,500)
    c = torch.FloatTensor(c)
    v_dataset = Data.TensorDataset(data_tensor=c, target_tensor = c)
    v_loader = Data.DataLoader(
        dataset = v_dataset,
        batch_size = 64,
        shuffle = False,
        num_workers=2
    )
    label = torch.cat((label,c),0)
    fake_a = torch.ones(1,103)
    for step, (_,batch_y) in enumerate(v_loader):
        # random.seed(step)
        noise.data.resize_(batch_y.size(0),40)
        noise.data.normal_(0, 1)
        batch_y = Variable(batch_y).cuda()
        noise_ = torch.cat((noise,batch_y),1)
        fake = netG(noise_)
        fake_a = torch.cat((fake_a,fake.data.cpu()),0)
    fake = fake_a[1:]
    fake_b = torch.cat((fake_b,fake),0)
num = arange(103)
print(fake_b[1:].numpy().shape)
print(label__[1:].shape)

data_b = '/media/jiming/_E/angelo/CGAN/Spectual.mat'
b = sio.loadmat(data_b)
data_all = b['x_spe_all']
label_all= b['t_label']



fake_ = fake_b[1:].numpy()
label__ = label__[1:]
t = np.vstack((fake_,x_spe))
s_label = np.ravel(s_label)
print(s_label.shape)
t_l = np.hstack((label__,s_label))


def create_svm(dataMat, dataLabel, decision='ovr'):
    clf = svm.SVC(C=100,gamma=0.01,decision_function_shape=decision)
    clf.fit(dataMat, dataLabel)
    return clf

model = create_svm(t,t_l)
preResult = model.predict(data_all)
score = model.score(data_all, label_all)

print("test feature1")
print("test dataMat shape: {0}, test dataLabel len: {1} ".format(data_all.shape, len(label_all)))
print("score: {:.6f}.".format(score))
print("error rate is {:.6f}.".format((1 - score)))
print("---------------------------------------------------------")

for i in arange(1,300):
    plt.figure(1)
    plt.plot(num,fake[i])

show()

for i in arange(1,30):
    plt.figure(2)
    plt.plot(num,data[i+30].numpy())
show()
