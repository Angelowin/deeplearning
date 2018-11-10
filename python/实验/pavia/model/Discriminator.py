import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,nc,ndf):
        super(Discriminator,self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf),
                                 nn.LeakyReLU(0.2,inplace=True))
        #305*170
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 152*85
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=5,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        #76*42
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=5,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*8),
                                 nn.LeakyReLU(0.2,inplace=True))
        #38*21
        self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,ndf*16,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*16),
                                 nn.LeakyReLU(0.2,inplace=True))
        #19*10
        self.layer6 = nn.Sequential(nn.Conv2d(ndf*16,ndf*32,kernel_size=5,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*32),
                                 nn.LeakyReLU(0.2,inplace=True))
        #9*5
        # self.layer7 = nn.Sequential(nn.Conv2d(ndf*32,ndf*64,kernel_size=(5,5),stride=(2,2),padding=(1,1)),
        #                          nn.BatchNorm2d(ndf*64),
        #                          nn.LeakyReLU(0.2,inplace=True))
        #4*2
        self.layer7 = nn.Sequential(nn.Conv2d(ndf*32,nc,kernel_size=4,stride=2,padding=0),
                                 nn.Sigmoid())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out
