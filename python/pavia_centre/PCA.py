# def data_proprecessing(data_U):
#     # minmaxscale = MinMaxScaler()
#     # minmaxscale.fit(data_U)
#     # data_U = minmaxscale.transform(np.array(data_U))
#
#     scale = StandardScaler()
#     scale.fit(data_U)
#     data_U = scale.transform(np.array(data_U))
#
#     pca = PCA( n_components= 3)
#     pca.fit(data_U)
#     data_U_pca = pca.transform(data_U)
#
#     return data_U, data_U_pca
#
# import scipy.io as sio
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split, KFold
# #from shift_between_twoset import SAD_matix, SSI
# import seaborn as sns
# import random
# import pandas as pd
#
# from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler
#
# from sklearn.decomposition import PCA
#
# data_U = sio.loadmat('G:\machinelearning\python\实验\data\Indian_pines_corrected.mat')[
#     'indian_pines_corrected']
# data_U = sio.loadmat('G:\machinelearning\python\pavia_centre\Pavia_centre.mat')['pavia']
# label_U = sio.loadmat('G:\machinelearning\python\pavia_centre\PaviaCentre_gt.mat')['pavia_gt']
# arr = np.array(label_U)
# mask = (arr ==9)
# arr_new = arr[mask]
# print(arr_new.size)
# label_U = sio.loadmat('G:\machinelearning\python\实验\data\Indian_pines_gt.mat')[
#     'indian_pines_gt']
# # data_U = sio.loadmat('D:\毕业论文\Code\data\PaviaU.mat')['paviaU']
# # label_U = sio.loadmat('D:\毕业论文\Code\data\PaviaU_gt.mat')['paviaU_gt']
#
# data_U = np.reshape(data_U, [145*145, 200], order = 'F')
# label_U = np.reshape(label_U, [145*145, ], order = 'F')
#
# label_ = label_U
#
# U_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#
# data_U, data_U_pca = data_proprecessing(data_U)
# data_U_pca = np.reshape(data_U_pca, [145, 145, 3], order = 'F')
# plt.xticks([])
# plt.yticks([])
# plt.imshow(data_U_pca)
# plt.show()




