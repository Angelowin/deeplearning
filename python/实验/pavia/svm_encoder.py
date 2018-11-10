import torch
import numpy as np
import scipy.io as sio
import svmutil as svm
import random
import sklearn.preprocessing as prep
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# label = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/ALofauto/PaviaU_gt.mat')['paviaU_gt']
data = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/ALofauto/PaviaU.mat')['paviaU']

data_500 = np.load('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_pavia/feature300_no10-1.npy')
# data_300 = np.load('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/Paper/degan_pavia/feature500_no2.npy')
# data_500 = np.vstack((data_500,data_300))
data_500 = torch.from_numpy(data_500)
data_500=torch.squeeze(data_500,dim=1).numpy()
data_500 = data_500.transpose(1,2,0)


#data = np.dstack((data_500,data_o))
# data = data[:,:,1]
# plt.imshow(data, cmap='gray')
# plt.show()
# data_o = sio.loadmat('/media/angelochen/E/MachineLearning/python/ALofauto/PaviaU.mat')['paviaU']
label = sio.loadmat('/media/jiming/5b343d7c-38ef-48d1-90c4-29c29b5df701/angelo/ALofauto/PaviaU_gt.mat')['paviaU_gt']
label = label[200:500,:300]



# data = data[200:500,:300,:]
# random.seed(10)
# index = [i for i in range(103)]
# random.shuffle(index)
# data_new =[]
# for i in range(80):
#     pic = data[:,:,i]
#     data_new.append(pic)
# data_new = np.array(data_new)
# data_new = data_new.transpose(1,2,0)

label = np.reshape(label,[300*300,1],order='F')



#label = [float(label[n]) for n in range(len(label)) if label[n] > 0]

# all_ind = np.array(label)
# print(all_ind.shape)
#
# for i in range(9):
#     x = label.count(i+1)
#     print(x)
data = np.reshape(data_500,[300*300,300],order = 'F')
data_p = prep.scale(data)
# pca = PCA(n_components=64)
# new_Data = pca.fit_transform(data_p)
data = [data_p[n] for n in range(len(label)) if label[n] > 0]
label = [float(label[n]) for n in range(len(label)) if label[n] >0]
data = np.array(data)
label = np.array(label)
random.seed(1)
index = [i for i in range(len(label))]
random.shuffle(index)
label_all = label[index]
x_all = data[index]

numt=200
# label_all = np.load('encoder_label64.npy')
# x_all = torch.load('feature64.pkl').data.numpy()
train_labels = label_all[:numt]
train_x = x_all[:numt,:]
test_x = x_all[numt:,:]
test_labels = label_all[numt:]


m = svm.svm_train(train_labels.tolist(), train_x.tolist(), '-c 200 -g 0.01 ')

tr_p_label, tr_p_acc, tr_p_val= svm.svm_predict(train_labels.tolist(),train_x.tolist(),m)
p_label, p_acc, p_val = svm.svm_predict(test_labels.tolist(), test_x.tolist(), m,)
print('debug')