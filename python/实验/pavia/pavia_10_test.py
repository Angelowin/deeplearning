import torch
import torchvision.utils as vutils
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import numpy as np
data = sio.loadmat('/media/jiming/_E/angelo/ALofauto/PaviaU.mat')['paviaU']

#data = data_o[200:500, :300, :]
random.seed(1)  # 1,5,40,50,100,500，550，600，700，750
index = [i for i in range(103)]
random.shuffle(index)
data_new = []
for i in range(10):
    pic = data[:, :, index[i]]
    data_new.append(pic)
    plt.show()
    data_new = np.array(data_new, dtype='float')
    data_new = torch.from_numpy(data_new)
    data_new = torch.unsqueeze(data_new, dim=1)
    # np.save('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/dcgan/s_image/600-test', data_new)  # 1,5,40,50,100,500,550,600,700,750
    vutils.save_image(data_new,
                      '/media/jiming/_E/angelo/Paper/degan_pavia/s_image/s_samples_1.png',
                      nrow=5, normalize=True)