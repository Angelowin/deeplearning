import matplotlib.pyplot as plt
import sklearn

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10,10)
y_sigmoid = 1/(1+np.exp(-x))
y_tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#fig = plt.figure()
# plot sigmoid
#ax = fig.add_subplot(221)
plt.figure(1)
plt.plot(x,y_sigmoid,color = "blue")
plt.grid()
#ax.set_title('Sigmoid')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plot tanh
plt.figure(2)
plt.plot(x,y_tanh,color = "blue")
plt.grid()

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plot relu
plt.figure(3)
y_relu = np.array([0*item  if item<0 else item for item in x ])
plt.plot(x,y_relu,color = "blue")
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plot leaky relu
plt.figure(4)
y_relu = np.array([0.2*item  if item<0 else item for item in x ])
plt.plot(x,y_relu,color = "blue")
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()

plt.show()