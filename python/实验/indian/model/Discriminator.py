import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,nc,ndf):
        super(Discriminator,self).__init__()
        # 32 x 32
        self.layer1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=5,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 72
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 36
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        #18
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*8),
                                 nn.LeakyReLU(0.2,inplace=True))
        #9
        self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,ndf*16,kernel_size=5,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*16),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 18
        self.layer6 = nn.Sequential(nn.Conv2d(ndf*16,1,kernel_size=4,stride=1,padding=0),
                                 nn.Sigmoid())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out
