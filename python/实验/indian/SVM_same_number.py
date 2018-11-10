#-*- coding: utf-8 -*-
import torch
import numpy as np
import scipy.io as sio
import svmutil as svm
import random
import sklearn.preprocessing as prep
#import hunxiao
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
#data_o = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Indian/Indian_pines_corrected.mat')['indian_pines_corrected']
#label = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Indian/Indian_pines_gt.mat')['indian_pines_gt']
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
from xlutils.copy import copy

import xlrd
import os
from xlutils.copy import copy
from xlwt import*


file = Workbook(encoding='utf-8')
table = file.add_sheet('indian_8-15')
file.save('indian_8-15')
def writeExcel(row, col, str, styl=Style.default_style):
    rb = xlrd.open_workbook('/media/jiming/_E/angelo/Paper/degan_indian/indian_8-15', formatting_info=True)
    wb = copy(rb)
    ws = wb.get_sheet(0)
    ws.write(row, col, str, styl)
    wb.save('/media/jiming/_E/angelo/Paper/degan_indian/indian_8-15')

data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/1.mat'
a = sio.loadmat(data)
data_tr1 = a["1"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/2.mat'
a = sio.loadmat(data)
data_tr2 = a["2"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/3.mat'
a = sio.loadmat(data)
data_tr3 = a["3"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/4.mat'
a = sio.loadmat(data)
data_tr4 = a["4"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/5.mat'
a = sio.loadmat(data)
data_tr5 = a["5"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/6.mat'
a = sio.loadmat(data)
data_tr6 = a["6"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/7.mat'
a = sio.loadmat(data)
data_tr7 = a["7"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/8.mat'
a = sio.loadmat(data)
data_tr8 = a["8"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/9.mat'
a = sio.loadmat(data)
data_tr9 = a["9"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/10.mat'
a = sio.loadmat(data)
data_tr10 = a["10"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/11.mat'
a = sio.loadmat(data)
data_tr11 = a["11"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/12.mat'
a = sio.loadmat(data)
data_tr12 = a["12"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/13.mat'
a = sio.loadmat(data)
data_tr13 = a["13"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/14.mat'
a = sio.loadmat(data)
data_tr14 = a["14"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/15.mat'
a = sio.loadmat(data)
data_tr15 = a["15"]


data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/16.mat'
a = sio.loadmat(data)
data_tr16 = a["16"]

data_tr=np.vstack((data_tr1,data_tr2,data_tr3,data_tr4,data_tr5,data_tr6,data_tr7,data_tr8,data_tr9,data_tr10,data_tr11,data_tr12,
           data_tr13,data_tr14,data_tr15,data_tr16))


# for i in range(16):
#     d = i
#     data = '/media/jiming/_E/angelo/Paper/degan_indian/bands/{}.mat'.format(d)
#     a = sio.loadmat(data)
#     data_tr_s = a[i]
#     label_tr_s = a[i]
#     np.vstack((label_tr,label_tr_s))

data_tr = prep.scale(data_tr)
##############


data_tr = np.array(data_tr)
data_tr = prep.scale(data_tr)
label_tr = np.array(label_tr)

data_tr = [data_tr[n] for n in range(len(label_tr)) if label_tr[n] > 0]

label_tr = [float(label_tr[n]) for n in range(len(label_tr)) if label_tr[n] > 0]
data_tr = np.array(data_tr)
label_tr = np.array(label_tr)

##################
##################
label = sio.loadmat('/media/jiming/_E/angelo/Indian/Indian_pines_gt.mat')[
    'indian_pines_gt']
data = sio.loadmat('/media/jiming/_E/angelo/Indian/Indian_pines_corrected.mat')[
    'indian_pines_corrected']
random.seed(40)  # 1,5,40,50,100,500,550,600,700,750
index = [i for i in range(200)]
random.shuffle(index)
plt.figure(1)
data_ne = []
for i in range(200):
    pic = data[:, :, i]
    # plt.subplot(2,5,i+1)
    # plt.axis('off')
    # plt.imshow(pic)

    data_ne.append(pic)

# data_new = np.array(data_ne,dtype='float')
# data_new = torch.from_numpy(data_new)
# data_new = torch.unsqueeze(data_new,dim=1)


data_new = np.transpose(data_ne, (1, 2, 0))
data_500 = np.array(data_new, dtype='float')
label = np.reshape(label, [145 * 145, 1], order='F')
data_p = np.reshape(data_500, [145 * 145, 200], order='F')
data_p = prep.scale(data_p)
##############



data_p = [data_p[n] for n in range(len(label)) if label[n] > 0]
index_x, index_y = np.where(label > 0)
label = [float(label[n]) for n in range(len(label)) if label[n] > 0]
data_p = np.array(data_p)
label = np.array(label)

al_aver = np.empty(shape=16)
train_acc = []
test_acc = []
####################
####################

kappa = []
for seed in range(1):
    random.seed(seed)
    index = [i for i in range(len(label))]
    random.shuffle(index)
    data_p_index = data_p[index]
    label_index = label[index]

    index2 = [i for i in range(len(label_tr))]
    random.shuffle(index2)
    data_p_tr = data_tr[index2]
    label_p_tr = label_tr[index2]

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

    # x =train_num[rand]  #103,205,308,410,513,1025

    label_train = label_p_tr[:]
    data_train = data_p_tr[:,:]


    test_x = data_p_index[:13000,:]
    test_labels = label_index[:13000]

    from xlwt import*
    m = svm.svm_train(label_train.tolist(), data_train.tolist(), '-c 200 -g 0.01 ')      #（400,0.05）
    # rand15 = 15*rand+a_all
    tr_p_label, tr_p_acc, tr_p_val= svm.svm_predict(label_train.tolist(),data_train.tolist(),m)
    p_label, p_acc, p_val = svm.svm_predict(test_labels.tolist(), test_x.tolist(), m,)
    #
    # hunxiao.plot_confusion_matrix(test_labels, p_label, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    #["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill","Soybean-mintill","Soybean-clean","Wheat","Woods","Buildings-Grass-Trees-Drives"
    #,"Stone-Steel-Towers"]
    # sorce_zeros = np.zeros(145*145)
    # for i in range(10249):
    #     sorce_zeros[index_x[i]] = p_label[i]
    #
    # label_new = np.reshape(sorce_zeros, [145,145], order='F')
    # label_new = label_new.astype(np.uint8)
    #
    # sio.savemat('s_indian30_40',mdict={'s_indian30':label_new})
#     writeExcel(rand15+seed+1, 0, str=tr_p_acc[0])
#     writeExcel(rand15+seed+1, 1, str=p_acc[0])
#     train_acc = np.hstack((train_acc,tr_p_acc[0]))
#     test_acc = np.hstack((test_acc,p_acc[0]))
#
#     ev_kappa = cohen_kappa_score(test_labels,p_label)
#     kappa = np.hstack((kappa,ev_kappa))
#
#     p_label = np.array(p_label)        # aver_aa = sum_aa / 16
#     # al_aver = np.vstack((al_aver,s_aa))
#     test_labels = np.array(test_labels)
#     sum_aa = 0
#     s_aa = []
#     for j in range(16):
#         num = 0
#         index2 = [n for n in range(len(test_labels)) if test_labels[n]==j+1]
#         for i in range(len(index2)):
#             if p_label[index2[i]]==test_labels[index2[i]]:
#                 num = num+1
#         aa = num/len(index2)
#         s_aa = np.hstack((s_aa,aa))
#         sum_aa = sum_aa+aa
#         writeExcel(rand15+seed+1,j+2,str=aa)
#         print('%.4f' %(num/len(index2)))
#
#     aver_aa = sum_aa / 16
#     al_aver = np.vstack((al_aver,s_aa))
#     # var = np.var(s_aa)
#     # std = np.std(s_aa)
#     # writeExcel(rand15+seed+1,j+3,aver_aa)
#     # writeExcel(rand15+seed+1,j+4,var)
#     # writeExcel(rand15+seed+1,j+5,std)
#
# kappa_end = np.average(kappa)
# kappa_std = np.std(kappa)
# print(kappa_end,kappa_std)
#
# writeExcel(0,0,str='train accuracy')
# writeExcel(0,1,str='test accuracy')
# # writeExcel(0,j+3,str='average')
# # writeExcel(0,j+4,str='var')
# # writeExcel(0,j+5,str='std')
# #writeExcel(seed+2,0,str='seed_aver')
# all = []
# for i in range(16):
#     aver = np.average(al_aver[1:11,i])
#     all = np.hstack((all,aver))
#     writeExcel(rand15+seed+2,i+2,str=aver)
# all_aver = np.average(all)
# all_var = np.var(all)
# all_std = np.std(all)
# train_acc = np.average(train_acc)
# test_acc = np.average(test_acc)
# writeExcel(rand15+seed+2,0,str=train_acc)
# writeExcel(rand15+seed+2,1,str=test_acc)
# # writeExcel(rand15+seed+2,i+3,all_aver)
# # writeExcel(rand15+seed+2,i+4,all_var)
# # writeExcel(rand15+seed+2,i+5,all_std)
# writeExcel(rand15+seed+2,i+4,kappa_end)
# writeExcel(rand15+seed+2,i+5,kappa_std)
# # writeExcel(rand15+seed+6,0,str='seed(%s)'%index_b)

