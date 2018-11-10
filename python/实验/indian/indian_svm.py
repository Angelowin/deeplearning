import torch
import numpy as np
import scipy.io as sio
import svmutil as svm
import random
import sklearn.preprocessing as prep
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

label = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Indian/Indian_pines_gt.mat')['indian_pines_gt']
data = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Indian/Indian_pines_corrected.mat')['indian_pines_corrected']
# label = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/ALofauto/PaviaU_gt.mat')['paviaU_gt']
# data = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/ALofauto/PaviaU.mat')['paviaU']

#data = np.reshape(data,[145*145,220],order='F')
#data_p = prep.scale(data)
# pca = PCA(n_components=3)
# # data = np.array(data)
# data = pca.fit_transform(data)
#data = np.reshape(data_p,[145,145,220],order='F')
# plt.imshow(data, cmap='gray')
# plt.show()
random.seed(1)
index = [i for i in range(200)]
random.shuffle(index)


data_new = []
for i in range(20):
    pic = data[:,:,index[i]]
    data_new.append(pic)

data_new = np.array(data_new)
data_new = data_new.transpose(1,2,0)

label = np.reshape(label,[145*145,1],order='F')
data = np.reshape(data_new,[145*145,20],order = 'F')
data_p = prep.scale(data)


# data_p = [data_p[n] for n in range(len(label)) if label[n] > 0]
# label = [float(label[n]) for n in range(len(label)) if label[n] >0]
# data_p = np.array(data_p)
# label = np.array(label)
# index = [i for i in range(len(label))]
# random.shuffle(index)
# data_p = data_p[index]
# label = label[index]
#
# i = 7
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
# train_labels = label_train1[index2]
# train_x = data_train1[index2]


# pca = PCA(n_components=64)
# new_Data = pca.fit_transform(data_p)
data = [data_p[n] for n in range(len(label)) if label[n] > 0]
label = [float(label[n]) for n in range(len(label)) if label[n] >0]
data = np.array(data)
label = np.array(label)
index = [i for i in range(len(label))]
random.shuffle(index)
label_all = label[index]
x_all = data[index]



train_labels = label_all[:1025]
train_x = x_all[:1025,:]

test_x = x_all[:13000,:]
test_labels = label_all[:13000]


m = svm.svm_train(train_labels.tolist(), train_x.tolist(), '-c 1300 -g 0.04 ')

tr_p_label, tr_p_acc, tr_p_val= svm.svm_predict(train_labels.tolist(),train_x.tolist(),m)
p_label, p_acc, p_val = svm.svm_predict(test_labels.tolist(), test_x.tolist(), m,)
print('debug')
