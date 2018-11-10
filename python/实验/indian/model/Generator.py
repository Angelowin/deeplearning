import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Generator,self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz,ngf*16,kernel_size=4),
                                 nn.BatchNorm2d(ngf*16),
                                 nn.ReLU())
        # 4 x 4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf*16,ngf*8,kernel_size=5,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*8),
                                 nn.ReLU())
        # 9
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf*8,ngf*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.ReLU())
        #18
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.ReLU())
        #36
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf),
                                 nn.ReLU())

        # 145
        self.layer6 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=5,stride=2,padding=1),
                                 nn.Tanh())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out
