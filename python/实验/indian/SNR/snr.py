import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import skimage
img = imread('/media/angelochen/E/MachineLearning/python/SNR/1803432755.jpg')
img1 = img[:,:,0]
img_gaussian = skimage.util.random_noise(img1, mode='salt', seed=None, clip=True)
#img = np.array(img)
plt.imshow(img1)
#plt.imshow(img_gaussian)
plt.show()
def psnr(im1,im2):
    diff = np.abs(im1 - im2)
    rmse = np.sqrt(diff).sum()
    psnr = 20*np.log10(255/rmse)
    return psnr



psnr_source = psnr(img1,img1)
psnr_gaussian = psnr(img1,img_gaussian)
print(psnr_source,psnr_gaussian)