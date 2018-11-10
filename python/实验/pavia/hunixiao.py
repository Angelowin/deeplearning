import matplotlib as mpl
import random

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

mpl.use('Agg')
#custom_font = mpl.font_manager.FontProperties(fname='/media/jiming/_E/angelo/Paper/degan_indian/SNR/zh.ttf')
def plot_confusion_matrix(y_true, y_pred, labels):
    # import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.get_cmap('Accent_r')
    # cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    intFlag = 1  # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='black', fontsize=15, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                # 这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=15, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='black', fontsize=15, va='center', ha='center')
    if (intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    # plt.title(u'ResNet模型预测混淆矩阵结果')
    plt.colorbar()
    # xy = range(0)
    # z = xy
    # sc = plt.scatter(z,z,c=z)
    # plt.colorbar(sc)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    # plt.ylabel(u'滚动轴承真实类别')
    # plt.xlabel(u'滚动轴承预测类别')
    plt.savefig('g_pavia_new30_5.jpg', dpi=300)
    plt.show()
