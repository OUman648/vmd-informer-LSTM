import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


if __name__ == '__main__':
    # 加载数据
    train_mse = load('train_mse')
    test_mse = load('test_mse')

    # 创建训练损失图
    plt.figure(figsize=(14, 10), dpi=300)

    plt.figure(figsize=(14, 7), dpi=100)  # dpi 越大  图片分辨率越高，写论文的话 一般建议300以上设置
    plt.plot(train_mse, label='Train MSE loss', marker='o', color='orange')
    plt.plot(test_mse, label='Test MSE loss', marker='*', color='green')
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.title('VMD + Informer-BiLSTM parallel training process Visualization', fontsize=16)
    # plt.show()  # 显示 lable
    plt.savefig('新训练图', dpi=100)