from joblib import dump, load
import numpy as np
import torch
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt


import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 模型 测试集 测试
def model_test(model, test_loader, pre_len):
    model = model.to(device)
    # 预测数据
    original_data = []
    pre_data = []
    # 每一个epoch结束后，在测试集上验证实验结果。
    with torch.no_grad():
        # 将模型设置为评估模式
        model.eval()
        for x, y, xt, yt in test_loader:
            # 创建一个掩码
            mask = torch.zeros_like(y)[:, -pre_len:, :].to(device)
            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            # print(y.size())  # torch.Size([64, 49, 19])
            # 覆盖掉未来信息
            dec_y = torch.cat([y[:, :-pre_len, :], mask], dim=1)
            # 前向传播
            test_pred = model(x, xt, dec_y[:, :, :-1], yt)  # torch.Size([64, 3, 1])
            # print(y_pred.size())
            # 损失计算
            # print(y[:, -pre_len:].size()) # torch.Size([64, 3, 1])
            # 使用 squeeze 移除尺寸为 1 的最后一个维度
            test_pred = test_pred.squeeze(-1)
            label = y[:, -pre_len:, -1]

            origin_lable = label.tolist()
            original_data += origin_lable

            test_pred = test_pred.tolist()
            pre_data += test_pred


    # 模型分数
    original_data = np.array(original_data)
    pre_data = np.array(pre_data)
    score = r2_score(original_data, pre_data)
    print('*' * 50)
    print('VMD + Informer-BiLSTM 模型分数--R^2:', score)

    print('*' * 50)
    # 测试集上的预测误差
    test_mse = mean_squared_error(original_data, pre_data)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(original_data, pre_data)
    print('测试数据集上的均方误差--MSE: ', test_mse)
    print('测试数据集上的均方根误差--RMSE: ', test_rmse)
    print('测试数据集上的平均绝对误差--MAE: ', test_mae)


    return original_data, pre_data




if __name__ == '__main__':

    # 参数设置  同训练设置
    pre_len = 1

    # 加载模型
    model = torch.load('best_model_vmd_informer_bilstm.pt')

    # 加载测试集
    test_loader = load('test_loader')

    # 模型预测
    original_data, pre_data = model_test(model, test_loader, pre_len)

    # 反归一化处理
    # 使用相同的均值和标准差对预测结果进行反归一化处理
    # 反标准化
    scaler = load('./data/scaler')
    original_data = scaler.inverse_transform(original_data)
    pre_data = scaler.inverse_transform(pre_data)

    # 可视化结果
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(original_data, label='true value', color='orange')  # 真实值
    plt.plot(pre_data, label='VMD + Informer-BiLSTM prediction', color='green')  # 预测值
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.title('VMD + Informer-BiLSTM parallel training process Visualization', fontsize=16)
    # plt.show()  # 显示 lable
    plt.savefig('预测拟合', dpi=100)
