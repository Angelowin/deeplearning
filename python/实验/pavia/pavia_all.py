#import torch
import numpy as np
import scipy.io as sio
#import svmutil as svm
import random
import sklearn.preprocessing as prep
import hunixiao
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

# data_test = data_o[:,:,12]
# random.seed(10)
# index = [i for i in range(200)]
# random.shuffle(index)
#
#
# data_new = []
# for i in range(20):
#     pic = data_o[:,:,i]
#     data_new.append(pic)
#
# data_new = np.array(data_new)
# data_new = data_new.transpose(1,2,0)
# data = data_new[2,:,:]
# plt.imshow(data,cmap = 'gray')
# plt.show()

# data_500 = np.load('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/feature100_no100-500.npy')
# data_500 = torch.from_numpy(data_500)
# data_500=torch.squeeze(data_500,dim=1).numpy()
# data_500 = data_500.transpose(1,2,0)
#
# #data = np.dstack((data_500,data_o))
# # data = data[:,:,1]
# # plt.imshow(data, cmap='gray')
# # plt.show()
# # data_o = sio.loadmat('/media/angelochen/E/MachineLearning/python/ALofauto/PaviaU.mat')['paviaU']
# # label = sio.loadmat('/media/angelochen/E/MachineLearning/python/ALofauto/PaviaU_gt.mat')['paviaU_gt']
#
# label = np.reshape(label,[145*145,1],order='F')
# data = np.reshape(data_500,[145*145,100],order = 'F')
# data_p = prep.scale(data)



from xlrd import open_workbook
#from xlutils.copy import copy

import xlrd
import os
#from xlutils.copy import copy
from xlwt import*


file = Workbook(encoding='utf-8')
table = file.add_sheet('pavia')
file.save('pavia')
# def writeExcel(row, col, str, styl=Style.default_style):
#     rb = xlrd.open_workbook('/media/jiming/_E/angelo/Paper/degan_pavia/pavia', formatting_info=True)
#     wb = copy(rb)
#     ws = wb.get_sheet(0)
#     ws.write(row, col, str, styl)
#     wb.save('/media/jiming/_E/angelo/Paper/degan_pavia/pavia')


    #style = easyxf('font:height 240, color-index red, bold on;align: wrap on, vert centre, horiz center');

# label = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Indian/Indian_pines_gt.mat')[
#         'indian_pines_gt']
# data_500 = np.load(
#         '/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/feature100_no10-50-test.npy')
# data_500 = torch.from_numpy(data_500)
# data_500 = torch.squeeze(data_500, dim=1).numpy()
# data_500 = data_500.transpose(1, 2, 0)
#
#
# for i in range(10):
#     pic = data_500[:,:,i]
#     plt.subplot(2, 5, i + 1)
#     plt.axis('off')
#     plt.imshow(pic)
#
# plt.show()

bands_index =[40] #[1,5,40,50,100,500,550,600,700,750]
for a in range(1):
    a_all = a*90
    index_b = bands_index[a]
    excel = 15
    train_num = [270]
    for rand in range(1):


    #############
        label = sio.loadmat('/media/jiming/_E/angelo/ALofauto/PaviaU_gt.mat')[
            'paviaU_gt']
        data = sio.loadmat('/media/jiming/_E/angelo/ALofauto/PaviaU.mat')[
            'paviaU']

        random.seed(index_b)  # 1,5,40,50,100,500,550,600,700,750
        index = [i for i in range(103)]
        random.shuffle(index)
        plt.figure(1)
        data_ne = []
        for i in range(103):
            pic = data[:, :, i]
            # plt.subplot(2,5,i+1)
            # plt.axis('off')
            # plt.imshow(pic)

            data_ne.append(pic)

        data_new = np.array(data_ne,dtype='float')
        data_new = torch.from_numpy(data_new)
        data_new = torch.unsqueeze(data_new,dim=1)


        data_new = np.transpose(data_ne,(1,2,0))
        data_500 = np.array(data_new, dtype='float')
        label = np.reshape(label, [610 * 340, 1], order='F')
        data_p = np.reshape(data_500, [610 * 340, 103], order='F')
        data_p = prep.scale(data_p)
    ##############

        data_p = [data_p[n] for n in range(len(label)) if label[n] > 0]
        index_x, index_y = np.where(label > 0)
        label = [float(label[n]) for n in range(len(label)) if label[n] >0]
        data_p = np.array(data_p)
        label = np.array(label)

        al_aver = np.empty(shape=9)
        train_acc =[]
        test_acc =[]


        kappa = []
        for seed in range(1):
            random.seed(seed)
            index = [i for i in range(len(label))]
            random.shuffle(index)
            data_p_index = data_p[index]
            label_index = label[index]
            #
            # i = 20
            # label_a = [label[n] for n in range(len(label)) if label[n]==1]
            # label_a = label_a[:i]
            # data_a = [data_p[n] for n in range(len(label)) if label[n]==1]
            # data_a = np.array(data_a)
            # data_a = data_a[:i,:]
            # for j in range(15):
            #     d1 = [data_p[m] for m in range(len(label)) if label[m]==j+2]
            #     l1 = [label[m] for m in range(len(label)) if label[m]==j+2]
            #     d1 = np.array(d1)
            #     #l1 = np.array(l1)
            #     l1 = l1[:i]
            #     d1 = d1[:i,:]
            #     label_a = np.append(label_a,l1)
            #     data_a = np.vstack((data_a,d1))



            # data_train1 = np.array(data_a)
            # label_train1 = np.array(label_a)
            # index2 = [i for i in range(len(label_train1))]
            # random.shuffle(index2)
            # label_train = label_train1[index2]
            # data_train = data_train1[index2]

            x =train_num[rand]  #103,205,308,410,513,1025

            label_train = label_index[:x]
            data_train = data_p_index[:x,:]


            test_x = data_p[:,:]
            test_labels = label[:]

            from xlwt import*
            m = svm.svm_train(label_train.tolist(), data_train.tolist(), '-c 200 -g 0.01 ')      #（400,0.05）
            rand15 = 15*rand+a_all
            tr_p_label, tr_p_acc, tr_p_val= svm.svm_predict(label_train.tolist(),data_train.tolist(),m)
            p_label, p_acc, p_val = svm.svm_predict(test_labels.tolist(), test_x.tolist(), m,)
            hunixiao.plot_confusion_matrix(test_labels, p_label, [1, 2, 3, 4, 5, 6, 7, 8, 9])
