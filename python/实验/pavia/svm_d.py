import torch
import numpy as np
import scipy.io as sio
import svmutil as svm
import random
import sklearn.preprocessing as prep
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
import torchvision.utils as vutils




from xlrd import open_workbook
from xlutils.copy import copy

import xlrd
import os
from xlutils.copy import copy
from xlwt import*


file = Workbook(encoding='utf-8')
table = file.add_sheet('pavia')
file.save('pavia')
def writeExcel(row, col, str, styl=Style.default_style):
    rb = xlrd.open_workbook('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_pavia/pavia', formatting_info=True)
    wb = copy(rb)
    ws = wb.get_sheet(0)
    ws.write(row, col, str, styl)
    wb.save('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_pavia/pavia')




bands_index = [1,5,40,50,100,500,550,600,700,750]
for a in range(10):
    a_all = a*90
    index_b = bands_index[a]
    excel = 15
    train_num = [1743,872,697,523,349,175]  #1743
    for rand in range(6):
        label = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/ALofauto/PaviaU_gt.mat')[
            'paviaU_gt']
        data_o = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/ALofauto/PaviaU.mat')['paviaU']
        data = data_o[200:500, :300, :]

        random.seed(index_b)  # 1,5,40,50,100,500，550，600，700，750
        index = [i for i in range(103)]
        random.shuffle(index)
        data_new = []
        for i in range(2):
            pic = data[:, :, index[i]]
            data_new.append(pic)

######show_begin
        plt.show()
        data_new = np.array(data_new, dtype='float')
        data_new = torch.from_numpy(data_new)
        data_new = torch.unsqueeze(data_new, dim=1)
        # np.save('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/dcgan/s_image/600-test', data_new)  # 1,5,40,50,100,500,550,600,700,750
        vutils.save_image(data_new,
                          '/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_indian/dcgan/s_image/s_samples.png',
                          nrow=5, normalize=True)

########show_end

        data_new = np.transpose(data_new, (1, 2, 0))
        data_p = np.reshape(data_new, [300 * 300, 40], order='F')
        data_p = prep.scale(data_p)


        label = label[200:500, :300]
        label = np.reshape(label, [300 * 300, 1], order='F')




        data_p = [data_p[n] for n in range(len(label)) if label[n] > 0]
        label = [float(label[n]) for n in range(len(label)) if label[n] >0]
        data_p = np.array(data_p)
        label = np.array(label)

        al_aver = np.empty(shape=9)
        train_acc =[]
        test_acc =[]


        kappa = []
        for seed in range(10):
            random.seed(seed)
            index = [i for i in range(len(label))]
            random.shuffle(index)
            data_p = data_p[index]
            label = label[index]


            x =train_num[rand]  #103,205,308,410,513,1025

            label_train = label[:x]
            data_train = data_p[:x,:]


            test_x = data_p[x:,:]
            test_labels = label[x:]

            from xlwt import*
            m = svm.svm_train(label_train.tolist(), data_train.tolist(), '-c 500 -g 0.2 ')      #（400,0.05）
            rand15 = 15*rand+a_all
            tr_p_label, tr_p_acc, tr_p_val= svm.svm_predict(label_train.tolist(),data_train.tolist(),m)
            p_label, p_acc, p_val = svm.svm_predict(test_labels.tolist(), test_x.tolist(), m,)
            writeExcel(rand15+seed+1, 0, str=tr_p_acc[0])
            writeExcel(rand15+seed+1, 1, str=p_acc[0])
            train_acc = np.hstack((train_acc,tr_p_acc[0]))
            test_acc = np.hstack((test_acc,p_acc[0]))

            ev_kappa = cohen_kappa_score(test_labels,p_label)
            kappa = np.hstack((kappa,ev_kappa))

            p_label = np.array(p_label)        # aver_aa = sum_aa / 16
            # al_aver = np.vstack((al_aver,s_aa))
            test_labels = np.array(test_labels)
            sum_aa = 0
            s_aa = []
            for j in range(9):
                num = 0
                index2 = [n for n in range(len(test_labels)) if test_labels[n]==j+1]
                for i in range(len(index2)):
                    if p_label[index2[i]]==test_labels[index2[i]]:
                        num = num+1
                aa = num/len(index2)
                s_aa = np.hstack((s_aa,aa))
                sum_aa = sum_aa+aa
                writeExcel(rand15+seed+1,j+2,str=aa)
                print('%.4f' %(num/len(index2)))

            aver_aa = sum_aa / 9
            al_aver = np.vstack((al_aver,s_aa))


        kappa_end = np.average(kappa)
        kappa_std = np.std(kappa)
        print(kappa_end,kappa_std)

        writeExcel(0,0,str='train accuracy')
        writeExcel(0,1,str='test accuracy')

        all = []
        for i in range(9):
            aver = np.average(al_aver[1:11,i])
            all = np.hstack((all,aver))
            writeExcel(rand15+seed+2,i+2,str=aver)
        all_aver = np.average(all)
        all_var = np.var(all)
        all_std = np.std(all)
        train_acc = np.average(train_acc)
        test_acc = np.average(test_acc)
        writeExcel(rand15+seed+2,0,str=train_acc)
        writeExcel(rand15+seed+2,1,str=test_acc)

        writeExcel(rand15+seed+2,i+4,kappa_end)
        writeExcel(rand15+seed+2,i+5,kappa_std)
        writeExcel(rand15+seed+6,0,str='seed(%s)'%index_b)