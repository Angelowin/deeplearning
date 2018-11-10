import random
import scipy.io as sio
import numpy as np
import torch
import svmutil as svm
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt


import xlrd
import os
from xlutils.copy import copy
from xlwt import*


file = Workbook(encoding='utf-8')
table = file.add_sheet('bbb')
file.save('bbb')
def writeExcel(row, col, str, styl=Style.default_style):
    rb = xlrd.open_workbook('/media/jiming/_E/angelo/Paper/degan_indian/bbb', formatting_info=True)
    wb = copy(rb)
    ws = wb.get_sheet(0)
    ws.write(row, col, str, styl)
    wb.save('/media/jiming/_E/angelo/Paper/degan_indian/bbb')


excel = 15
train_num = [1025]
for rand in range(1):
    label = sio.loadmat('/media/jiming/_E/angelo/Indian/Indian_pines_gt.mat')[
        'indian_pines_gt']
    data = sio.loadmat('/media/jiming/_E/angelo/Indian/Indian_pines_corrected.mat')[
        'indian_pines_corrected']

    random.seed(1)  # 1,5,40,50,100,500
    index = [i for i in range(200)]
    random.shuffle(index)

    data_new = []
    for i in range(30):
        pic = data[:, :, index[i]]
        data_new.append(pic)
    data_500 = np.array(data_new, dtype='float')
    label = np.reshape(label, [145 * 145, 1], order='F')
    data_p = np.reshape(data_500, [145 * 145, 30], order='F')
    data_p = prep.scale(data_p)


    data_p = [data_p[n] for n in range(len(label)) if label[n] > 0]
    label = [float(label[n]) for n in range(len(label)) if label[n] >0]
    data_p = np.array(data_p)
    label = np.array(label)

    al_aver = np.empty(shape=16)
    train_acc =[]
    test_acc =[]


    kappa = []
    for seed in range(1):
        random.seed(seed)
        index2 = [i for i in range(len(label))]
        random.shuffle(index2)
        data_p = data_p[index2]
        label = label[index2]
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

        label_train = label[:x]
        data_train = data_p[:x,:]


        test_x = data_p[x:13000,:]
        test_labels = label[x:13000]

        from xlwt import*
        m = svm.svm_train(label_train.tolist(), data_train.tolist(), '-c 500 -g 0.05 ')
        rand15 = 15*rand
        tr_p_label, tr_p_acc, tr_p_val= svm.svm_predict(label_train.tolist(),data_train.tolist(),m)
        p_label, p_acc, p_val = svm.svm_predict(test_labels.tolist(), test_x.tolist(), m,)
        writeExcel(rand15+seed+1, 0, str=tr_p_acc[0])
        writeExcel(rand15+seed+1, 1, str=p_acc[0])
