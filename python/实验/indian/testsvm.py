import random
import scipy.io as sio
import numpy as np
import torch
import svmutil as svm
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
import xlrd
import os
from xlutils.copy import copy
from xlwt import*

###########用于gabor_begin

# data_500 = np.load(
#     '/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/feature100_no10-750.npy')
# data_500 = torch.from_numpy(data_500)
# data_500 = torch.squeeze(data_500, dim=1).numpy()
# data_500 = data_500.transpose(1, 2, 0)
#
# sio.savemat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/xgabor/10xgabor/10-100-750.mat',{'no750':data_500})
# # for i in range(10):
# #     data_show= data_500[:,:,i]
# #     plt.imshow(data_show)
# #     plt.show()

data = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Indian/Indian_pines_corrected.mat')[
    'indian_pines_corrected']

random.seed(750)  # 1,5,40,50,100,500
index = [i for i in range(200)]
random.shuffle(index)
plt.figure(1)
data_ne = []
for i in range(10):
    pic = data[:, :, index[i]]

    data_ne.append(pic)


data_new = np.transpose(data_ne, (1, 2, 0))
sio.savemat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/xgabor/10xgabor/10-750.mat',{'n750':data_new})



##########用于gabor_end


file = Workbook(encoding='utf-8')
table = file.add_sheet('bbb')
file.save('bbb')
def writeExcel(row, col, str, styl=Style.default_style):
    rb = xlrd.open_workbook('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/bbb', formatting_info=True)
    wb = copy(rb)
    ws = wb.get_sheet(0)
    ws.write(row, col, str, styl)
    wb.save('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/bbb')

excel = 15
train_num = [1000,513,410,308,205,103]
for rand in range(6):
    label = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Indian/Indian_pines_gt.mat')[
        'indian_pines_gt']
    data = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Indian/Indian_pines_corrected.mat')[
        'indian_pines_corrected']

    random.seed(50)  # 1,5,40,50,100,500
    index = [i for i in range(200)]
    random.shuffle(index)
    plt.figure(1)
    data_ne = []
    for i in range(10):
        pic = data[:, :, index[i]]
        plt.subplot(2,5,i+1)
        plt.axis('off')
        plt.imshow(pic)
        # plt.imshow(pic)
        # plt.show()
        data_ne.append(pic)

    plt.show()
    data_new = np.array(data_ne,dtype='float')
    data_new = torch.from_numpy(data_new)
    data_new = torch.unsqueeze(data_new,dim=1)
    #np.save('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/dcgan/s_image/600-test', data_new)  # 1,5,40,50,100,500,550,600,700,750
    vutils.save_image(data_new,
                      '/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/dcgan/s_image/s_samples.png',nrow=5,normalize=True)

    data_new = np.transpose(data_ne,(1,2,0))
    data_500 = np.array(data_new, dtype='float')
    label = np.reshape(label, [145 * 145, 1], order='F')
    data_p = np.reshape(data_500, [145 * 145, 10], order='F')
    data_p = prep.scale(data_p)


    data_p = [data_p[n] for n in range(len(label)) if label[n] > 0]
    label = [float(label[n]) for n in range(len(label)) if label[n] >0]
    data_p = np.array(data_p)
    label = np.array(label)

    al_aver = np.empty(shape=16)
    train_acc =[]
    test_acc =[]


    kappa = []
    for seed in range(10):
        random.seed(seed)
        index2 = [i for i in range(len(label))]
        random.shuffle(index2)
        data_p = data_p[index2]
        label = label[index2]
        x =train_num[rand]  #103,205,308,410,513,1025

        label_train = label[:x]
        data_train = data_p[:x,:]


        test_x = data_p[x:13000,:]
        test_labels = label[x:13000]

        from xlwt import*
        m = svm.svm_train(label_train.tolist(), data_train.tolist(), '-c 800 -g 0.7 ')
        rand15 = 15*rand
        tr_p_label, tr_p_acc, tr_p_val= svm.svm_predict(label_train.tolist(),data_train.tolist(),m)
        p_label, p_acc, p_val = svm.svm_predict(test_labels.tolist(), test_x.tolist(), m,)
        writeExcel(rand15+seed+1, 0, str=tr_p_acc[0])
        writeExcel(rand15+seed+1, 1, str=p_acc[0])
