import os
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from vmdpy import VMD  # 确保已安装vmdpy库

# 数据读取，预处理
def data_preprocessing(dir, filename, target):
    '''
    参数
    :param dir: 文件路径
    :param filename: 数据文件名称 xxx.csv   第一列 一定是 时间！！！
    :param target: 预测的目标变量名称
    :return:
    df_normalized：归一化后的原始数据框
    target_values：归一化后的目标变量
    '''
    file_path =  os.path.join(dir, filename)
    original_data = pd.read_csv(file_path)

    # 数据集太大，缩小数据集处理！  如果数据集比较小，可以注释这行代码！
    # 由于数据集太大， 取前 6000条数据进行测试
    original_data = original_data.iloc[0:6000,:]   # original_data = original_data.iloc[:int(len(original_data)/6), ]


    # 分离时间列和其他列
    time_col = original_data.iloc[:, 0]  # 获取时间列
    data_cols = original_data.iloc[:, 1:]  # 获取其余列
    # 修改时间列的列名
    time_col = time_col.rename('date')

    # 拿出目标列对其进行处理
    target_values_col = original_data[target]
    target_values = np.array(target_values_col.tolist())  # 转换为numpy
    target_values = target_values.reshape(-1, 1)
    # 归一化处理
    # 使用标准化（z-score标准化）
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_cols)
    target_values = scaler.fit_transform(target_values)
    # 保存 归一化 模型
    dump(scaler, './data/scaler')

    # 将归一化后的数据合并回原始数据框
    normalized_df = pd.DataFrame(normalized_data, columns=data_cols.columns)
    df_normalized = pd.concat([time_col, normalized_df], axis=1)
    target_values = target_values.reshape(-1)
    df_normalized[target] = target_values

    return df_normalized


# 通过滑动窗口制作多步预测数据集
def create_multistep_dataset(data, window_size, label_len, forecast_step, task_type='MS'):
    '''
    参数：
    :param data: 数据元组（特征数据，标签数据）， 单变量--（特征数据，）
    :param window_size:  样本窗口大小
    :param label_len:    # Informer 解码器的起始 token 长度, decoder中 输入的没有掩码部分序列长度
    :param forecast_step: 多步预测 步数
    :param task_type:  任务类型(数据格式：字符串类型)  S：单变量预测单变量，MS：多变量预测单变量，默认 'MS'
    :return:
        sample_features  : 特征数据
        labels           : 签数据
    '''
    sample_features = []
    labels = []

    # 第一种任务 MS：多变量预测单变量
    if task_type == 'MS':
        features = data.values
        ylabel = data.values
        for i in range(len(ylabel) - window_size - forecast_step + 1):
            sample_features.append(features[i:i + window_size, :])
            labels.append(ylabel[i + window_size - label_len:i + window_size + forecast_step, :])

    # 第二种任务 S：单变量预测单变量
    elif task_type == 'S':
        features = data.values  # (24528, 1)
        for i in range(len(features) - window_size - forecast_step + 1):
            sample_features.append(features[i:i + window_size, :])
            labels.append(features[i + window_size- label_len:i + window_size + forecast_step, :])

    # 将列表转换为单一的NumPy数组
    sample_features = np.array(sample_features)
    labels = np.array(labels)

    return sample_features, labels


# 制作多步预测数据集
def make_dataset(df_normalized, target, window_size, label_len, forecast_step, task_type='MS', split_rate=[0.7, 0.3]):
    '''
    参数
    :param df_normalized:  归一化后的 CSV 数据！
    :param window_size:    数据滑动窗口值
    :param label_len:    # Informer 解码器的起始 token 长度, decoder中 输入的没有掩码部分序列长度
    :param forecast_step:  预测步数
    :param task_type: 任务类型(数据格式：字符串类型)  S：单变量预测单变量，MS：多变量预测单变量，默认 'MS'
    :param split_rate:     数据划分比例
    :return:
           train_xdata: 训练集数据
           train_ylabel: 训练集标签
           test_xdata: 测试集数据
           test_ylabel: 测试集标签
    '''
    # 第一步，划分数据集
    sample_len = df_normalized.shape[0]  # 样本总长度
    train_len = int(sample_len * split_rate[0])  # 向下取整

    # 第一种任务 MS：多变量预测单变量
    if task_type == 'MS':
        train_data = df_normalized.iloc[:train_len, :]  # 训练集
        test_data = df_normalized.iloc[train_len:, :]  # 测试集

        # 第二步，制作数据集标签  滑动窗口
        train_xdata, train_ylabel = create_multistep_dataset(train_data, window_size, label_len, forecast_step, task_type)
        test_xdata, test_ylabel = create_multistep_dataset(test_data, window_size, label_len, forecast_step, task_type)

    # 第二种任务 S：单变量预测单变量
    elif task_type == 'S':
        target_values = df_normalized[['date', target]]
        train_data = target_values.iloc[:train_len, :]  # 训练集 标签
        test_data = target_values.iloc[train_len:, :]  # 训练集 标签
        # 第二步，制作数据集标签  滑动窗口
        train_xdata, train_ylabel = create_multistep_dataset(train_data, window_size, label_len, forecast_step, task_type)
        test_xdata, test_ylabel = create_multistep_dataset(test_data, window_size, label_len, forecast_step, task_type)

    # 参数错误
    else:
        print("task_type ERROR!")
        return

    return train_xdata, train_ylabel, test_xdata, test_ylabel


# VMD 样本分解
def Sample_decomposition(num_imfs, dataset, forecast_step=1, mask=False):
    '''
       参数
       :param num_imfs:  vmd 分解分量
       :param dataset:   待分解的数据集
       :param forecast_step:  预测步数
       :param mask:    是否创建掩码
       :return:
              decomposed_data: 分解后的数据
       '''
    datetime_col, numeric_data = preprocess_data(dataset)

    # VMD 参数设置
    # alpha 惩罚系数；带宽限制经验取值为抽样点长度1.5-2.0倍.
    # 惩罚系数越小，各IMF分量的带宽越大，过大的带宽会使得某些分量包含其他分量言号;
    alpha = 2000
    tau = 0.
    # 模态数量  分解模态（IMF）个数
    K = num_imfs
    # 0表示不是DC分量，1表示是DC分量
    DC = 0
    # 初始化
    init = 1
    # 终止条件
    tol = 1e-7
    # ---------------

    # 处理 编码器 输入
    # 处理思路：把目标变量那一列进行分解，然后把分解后的分量替换原始目标变量，合并到原始数据集中
    if mask is False:
        # 初始化新的数据矩阵, 最终的特征数应该是 13 + K = 21 (先不包括时间列 和标签列 15-1-1 + k )
        decomposed_data = np.zeros((numeric_data.shape[0], numeric_data.shape[1], numeric_data.shape[2] - 1 + K))
        # 遍历每个样本，并对目标变量进行VMD分解
        for i in range(numeric_data.shape[0]):
            # 提取当前样本的原始特征 (前13个特征)
            original_features = numeric_data[i, :, :-1]  # shape: (96, 13)

            # 提取目标变量（最后一个特征）
            target_signal = numeric_data[i, :, -1]

            # 进行VMD分解
            # 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
            u, u_hat, omega = VMD(target_signal, alpha, tau, K, DC, init, tol)

            # 将分解后的分量添加到原始特征
            # u 的形状为 (K, 96)，我们需要将其转置为 (96, K) 后才能拼接
            vmd_components = u.T  # shape: (96, K)

            # 将原始特征和分解后的分量进行拼接
            new_sample = np.concatenate((original_features, vmd_components), axis=1)  # shape: (96, 21)

            # 将更新后的样本放入新的数据集中
            decomposed_data[i] = new_sample

    # 处理 解码器 输入
    # 处理思路：把目标变量那一列拿出来，遮掩住预测部分标签值，进行分解，然后把分解后的分量合并到原始数据集中，原始目标变量放在最后一列
    else:
        # 初始化新的数据矩阵, 最终的特征数应该是 14 + K = 22 (先不包括时间列 和标签列 15-1 + k )
        decomposed_data = np.zeros((numeric_data.shape[0], numeric_data.shape[1], numeric_data.shape[2]+ K))
        # 遍历每个样本，并对目标变量进行VMD分解
        for i in range(numeric_data.shape[0]):
            # 提取当前样本的原始特征 (前13个特征)
            original_features = numeric_data[i, :, :-1]  # shape: (49, 13)

            # 提取目标变量（最后一个特征）
            target_signal = numeric_data[i, :, -1]
            # 创建一个 掩码

            # 创建一个 掩码, 这里需要使用 .copy() 创建 target_signal 的副本
            mask_target_signal = target_signal.copy()
            # 将最后 forecast_step 个位置设置为 0
            mask_target_signal = mask_target_signal[:-forecast_step]

            # 进行VMD分解
            # 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
            u, u_hat, omega = VMD(mask_target_signal, alpha, tau, K, DC, init, tol)

            # 将分解后的分量添加到原始特征
            # u 的形状为 (K, 48)，我们需要将其转置为 (48, K) 后才能拼接
            vmd_components = u.T  # shape: (48, K)
            # 创建 (forecast_step, K) 的全为 0 的矩阵
            matrix_mask = np.zeros((forecast_step, K))
            # 合并两个矩阵
            vmd_components = np.vstack((vmd_components, matrix_mask)) # shape: (49, K)

            # 将原始特征和分解后的分量进行拼接
            new_sample = np.concatenate((original_features, vmd_components), axis=1)  # shape: (49, 21)
            # 将 原始目标变量放在最后一列
            # 需要将 (49,) 的数据转化为 (49, 1) 的列向量
            target_signal = target_signal.reshape(numeric_data.shape[1], 1)
            new_sample = np.hstack((new_sample, target_signal))  # shape: (49, 19)

            # 将更新后的样本放入新的数据集中
            decomposed_data[i] = new_sample

    # 将时间列扩展为 (xxx, 96, 1)
    datetime_col_expanded = datetime_col[:, :, np.newaxis]  # shape (xxx, 96, 1)
    # 将时间列 (xxxx, 96, 1) 和 decomposed_data (xxxx, 96, 18) 拼接
    # 在 axis=2 维度上拼接，确保时间列在第一个位置
    decomposed_data = np.concatenate((datetime_col_expanded, decomposed_data), axis=2)

    return decomposed_data


# 处理数据集（带有日期时间列）
def preprocess_data(data):
    '''
    :param data: 包含日期时间字符串的数据集
    :return: 返回适合 VMD 分解的数值数据
    '''
    # 将日期时间列保留为特征
    datetime_col = data[:, :, 0]

    # 去除日期时间列并保留数值数据
    numeric_data = data[:, :, 1:]

    return datetime_col, numeric_data


# 自定义数据集
class MyData(Dataset):
    def __init__(self, values, labels):
        self.values, self.labels = values, labels

    def __len__(self):
        return len(self.values)

    def create_time(self, data):
        # 提取时间列
        time = data[:, 0]
        time = pd.to_datetime(time)

        # 提取各时间特征
        week = np.int32(time.dayofweek)[:, None]
        month = np.int32(time.month)[:, None]
        day = np.int32(time.day)[:, None]
        hour = np.int32(time.hour)[:, None]
        minute = np.int32(time.minute)[:, None]
        time_data = np.concatenate([month, day, week, hour, minute], axis=-1)

        return time_data

    def __getitem__(self, item):
        value = self.values[item]  # (12, 15)
        label = self.labels[item]  # (9, 2)

        value_t = self.create_time(value) # (12, 5)
        label_t = self.create_time(label) # (9, 5)

        value = value[:, 1:]
        label = label[:, 1:]

        value = np.float32(value)
        label = np.float32(label)
        return value, label, value_t, label_t


if __name__ == "__main__":

    # 数据集 文件路径
    dir ='./data/WindPower/'
    # 数据文件名称 xxx.csv
    filename = 'WindPower.csv'
    # 预测的目标变量名称
    target = '实际发电功率（mw）'
    # 数据读取，预处理
    df_normalized = data_preprocessing(dir, filename, target)
    print(df_normalized.shape)  # (6000, 15)
    # 定义序列长度和预测步数
    # 定义窗口大小  ： 用过去 96个步长 ，预测未来 1个 步长  (单步预测)
    window_size = 96
    # Informer 解码器的起始 token 长度, decoder中 输入的没有掩码部分序列长度
    label_len = 48
    # 预测步数
    forecast_step = 1
    # 数据集划分比例
    split_rate = [0.7, 0.3]

    # 任务类型 S：单变量预测单变量，MS：多变量预测单变量，默认 'MS'
    task_type = 'MS'

    # 制作数据集
    train_xdata, train_ylabel, test_xdata, test_ylabel = make_dataset(df_normalized, target, window_size, label_len,
                                                                      forecast_step, task_type)

    # print('数据 形状：')
    print(train_xdata.shape, train_ylabel.shape)  # (4104, 96, 15) (4104, 49, 15)
    print(test_xdata.shape, test_ylabel.shape)  # (1704, 96, 15) (1704, 49, 15)

    # VMD 分解  一定要先划分数据集！然后把数据和标签制作好，最后来对每个样本进行分解！！！
    print('*' * 20, '开始 VMD 分解', '*' * 20)
    # 设置分解分量数目
    num_imfs = 8
    # 编码器输入
    train_xdata = Sample_decomposition(num_imfs, train_xdata)
    test_xdata = Sample_decomposition(num_imfs, test_xdata)

    # 解码器输入  注意解码器输入，需要把 label_len 进行分解， 但是不分解 标签值！通过创建掩码解决
    train_ylabel = Sample_decomposition(num_imfs, train_ylabel, forecast_step=forecast_step, mask=True)
    test_ylabel = Sample_decomposition(num_imfs, test_ylabel, forecast_step=forecast_step, mask=True)

    print('*' * 20, 'VMD 分解结束', '*' * 20)
    # print('数据 形状：')
    print(train_xdata.shape, train_ylabel.shape)  # (4104, 96, 22) (4104, 49, 23)
    print(test_xdata.shape, test_ylabel.shape)  # (1704, 96, 22) (1704, 49, 23)

    # 保存数据集
    # 保存数据
    dump(train_xdata, './data/data-label/train_xdata')
    dump(test_xdata, './data/data-label/test_xdata')
    dump(train_ylabel, './data/data-label/train_ylabel')
    dump(test_ylabel, './data/data-label/test_ylabel')

    # 训练集
    # train_xdata = load('./data/data-label/train_xdata')
    # train_ylabel = load('./data/data-label/train_ylabel')
    # # 测试集
    # test_xdata = load('./data/data-label/test_xdata')
    # test_ylabel = load('./data/data-label/test_ylabel')
    #
    # train_data = MyData(train_xdata, train_ylabel)

    # for x, y in train_data:
    #     break