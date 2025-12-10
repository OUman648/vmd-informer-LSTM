import torch
from joblib import dump, load
import torch.nn as nn
import numpy as np
import time
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# 参数与配置
torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 有GPU先用GPU训练

# 导入模型定义
from make_dataset import MyData
from models.model import InformerBiLSTMModel


# 加载数据集
def dataloader(batch_size, workers=2):
    # 训练集
    train_xdata = load('./data/data-label/train_xdata')
    train_ylabel = load('./data/data-label/train_ylabel')
    # 测试集
    test_xdata = load('./data/data-label/test_xdata')
    test_ylabel = load('./data/data-label/test_ylabel')

    # 加载数据
    train_loader = Data.DataLoader(MyData(train_xdata, train_ylabel),
                                   batch_size=batch_size, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(MyData(test_xdata, test_ylabel),
                                  batch_size=batch_size, num_workers=workers, drop_last=True)
    return train_loader, test_loader



# 训练模型
def model_train(train_loader, test_loader, parameter):
    '''
          参数
          train_loader：训练集
          test_loader：测试集
          parameter： 参数
          返回
      '''
    device = parameter['device']
    model = parameter['model']
    model = model.to(device)
    # 参数
    epochs = parameter['epochs']
    learn_rate = parameter['learn_rate']
    pre_len = parameter['out_len'] # 预测长度

    # 定义损失函数和优化函数
    loss_function = nn.MSELoss() # loss
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)  # 优化器

    # 最低MSE
    minimum_mse = 1000.
    # 最佳模型
    best_model = model

    train_mse = []  # 记录在训练集上每个epoch的 MSE 指标的变化情况   平均值
    test_mse = []  # 记录在测试集上每个epoch的 MSE 指标的变化情况   平均值

    print('*' * 20, '开始训练', '*' * 20)
    # 计算模型运行时间
    start_time = time.time()
    for epoch in range(epochs):
        # 训练
        model.train()
        train_mse_loss = []  # 保存当前epoch的MSE loss和
        for x, y ,xt, yt in train_loader:
            # 创建一个掩码
            mask = torch.zeros_like(y)[:, -pre_len:, :].to(device) # torch.Size([64, 1, 19])
            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            # print(y.size())  # torch.Size([64, 49, 19])
            # 覆盖掉未来信息
            dec_y = torch.cat([y[:, :-pre_len, :], mask], dim=1)
            # 每次更新参数前都梯度归零和初始化
            optimizer.zero_grad()
            # 前向传播
            y_pred = model(x, xt, dec_y[:, :, :-1], yt)  # torch.Size([64, 1, 1])
            # print(y_pred.size())
            # 损失计算
            # 使用 squeeze 移除尺寸为 1 的最后一个维度
            y_pred = y_pred.squeeze(-1) # torch.Size([64, 1])
            label = y[:, -pre_len:, -1]
            loss = loss_function(y_pred, label)
            train_mse_loss.append(loss.item())  # 计算 MSE 损失
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
        #     break
        # break
        # 计算总损失
        train_av_mseloss = np.average(train_mse_loss)  # 平均
        train_mse.append(train_av_mseloss)

        print(f'Epoch: {epoch + 1:2} train_MSE-Loss: {train_av_mseloss:10.8f}')
        # 每一个epoch结束后，在测试集上验证实验结果。
        with torch.no_grad():
            # 将模型设置为评估模式
            model.eval()
            test_mse_loss = []  # 保存当前epoch的MSE loss和
            for x, y, xt, yt in test_loader:
                # 创建一个掩码
                mask = torch.zeros_like(y)[:, -pre_len:, :].to(device)
                x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
                # 覆盖掉未来信息
                dec_y = torch.cat([y[:, :-pre_len, :], mask], dim=1)
                # 每次更新参数前都梯度归零和初始化
                optimizer.zero_grad()
                # 前向传播
                y_pred = model(x, xt, dec_y[:, :, :-1], yt)  # torch.Size([64, 1, 1])
                # print(y_pred.size())
                # 损失计算
                # 使用 squeeze 移除尺寸为 1 的最后一个维度
                y_pred = y_pred.squeeze(-1)
                label = y[:, -pre_len:, -1]
                test_loss = loss_function(y_pred, label)
                test_mse_loss.append(test_loss.item())

            # 计算总损失
            test_av_mseloss = np.average(test_mse_loss)  # 平均
            test_mse.append(test_av_mseloss)
            print(f'Epoch: {epoch + 1:2} test_MSE_Loss:{test_av_mseloss:10.8f}')
            # 如果当前模型的 MSE 低于于之前的最佳准确率，则更新最佳模型
            # 保存当前最优模型参数
            if test_av_mseloss < minimum_mse:
                minimum_mse = test_av_mseloss
                best_model = model  # 更新最佳模型的参数

    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    # 最后的模型参数
    last_model = model
    print('*' * 20, '训练结束', '*' * 20)
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    print(f'min_MSE: {minimum_mse}')

    # 可视化
    # 创建训练损失、准确率图
    plt.figure(figsize=(14, 7), dpi=100)  # dpi 越大  图片分辨率越高，写论文的话 一般建议300以上设置
    plt.plot(range(epochs), train_mse, label='Train MSE loss', marker='o', color='orange')
    plt.plot(range(epochs), test_mse, label='Test MSE loss', marker='*', color='green')
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.title('VMD+Informer-BiLSTM parallel training process Visualization', fontsize=16)
    #plt.show()  # 显示 lable
    plt.savefig('train_result', dpi=100)
    # 保存结果 方便 后续画图处理（如果有需要的话）
    dump(train_mse, 'train_mse')
    dump(test_mse, 'test_mse')
    return last_model, best_model


if __name__ == '__main__':
    batch_size = 64
    # 加载数据
    train_loader, test_loader = dataloader(batch_size)
    # 保存测试集数据， 后面进行测试
    dump(test_loader, 'test_loader')

    # BiLSTM 隐藏层 两层， 每次分别 32, 64个神经元
    hidden_layer_sizes = [32, 64]
    # 编码器输入大小（默认为7） 输入数据的特征数量，不包括时间的那一列！
    enc_in = 21  # 咱们数据集 除去时间和被分解的特征，一共 13个特征 + 8 个分量
    # 解码器输入大小  同编码器
    dec_in = 21
    # 输出数据维度大小 咱们的任务都是预测单变量， 所以维度为1
    c_out = 1
    #  窗口大小 window_size 大小
    seq_len =  96
    # Informer 解码器的起始 token 长度, decoder中 输入的没有掩码部分序列长度
    label_len = 48
    # 预测长度
    out_len = 1
    #  Probesparse attn因子（默认为5）
    factor = 5
    # 模型维度
    d_model = 128
    # 多头注意力头数
    n_heads = 4
    # 编码器层数 默认 2 层
    e_layers = 1
    # 解码器层数 默认 1 层
    d_layers = 1
    # 模型中全连接网络（FCN）的维度
    d_ff = 200
    # dropout概率
    dropout = 0.1
    # 时间特征编码的频率（默认为h）。可以将其设置为s、t、h、d、b、w、m
    # （s：其次，t：每分钟，h：每小时，d：每天，b：工作日，w：每周，m：每月）。
    # 你也可以使用更详细的频率，如15分钟或3小时
    # 我们的数据为分钟数据
    freq = 't'

    # 一下参数默认即可
    # 编码器中使用的注意事项（默认为prob） 默认为"prob"论文的主要改进点，提出的注意力机制
    attn='prob'
    # 时间特征编码（默认为timeF）
    embed='fixed'
    # 激活函数（默认为gelu）
    activation='gelu'
    output_attention = False
    distil = True
    mix = True,
    # 定义 InformerBiLSTMModel 模型
    model = InformerBiLSTMModel(hidden_layer_sizes, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor, d_model, n_heads, e_layers, d_layers, d_ff,dropout, attn, embed, freq, activation,output_attention,distil, mix)

    # 训练 参数设置
    learn_rate = 0.0003  # 学习率
    epochs = 50

    # 制作参数字典
    parameter = {
        'model': model,
        'epochs': epochs,
        'learn_rate': learn_rate,
        'out_len': out_len,
        'device':device
    }

    # 训练模型
    last_model, best_model = model_train(train_loader, test_loader, parameter)
    # 保存最后的参数
    # torch.save(last_model, 'final_model_vmd_informer_bilstm.pt')
    # 保存最好的参数
    torch.save(best_model, 'best_model_vmd_informer_bilstm.pt')