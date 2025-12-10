import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

'''
Informer
   - Informer是一种单层的模型结构，它由编码器和解码器组成，适用于处理单层的时间序列预测任务。
   - 其编码器和解码器部分都包含了自注意力机制和多头注意力机制，用于捕捉时间序列数据中的长距离依赖关系。
   - 适用于一般的时间序列预测任务，可以处理单个时间序列的预测。
'''

class InformerBiLSTMModel(nn.Module):
    def __init__(self, hidden_layer_sizes, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        '''
        :param hidden_layer_sizes:  BiLSTM 隐藏层的数目和维度
        :param enc_in: 编码器输入大小, 输入数据的特征数量，不包括时间的那一列！
        :param dec_in: 解码器输入大小，输入数据的特征数量，不包括时间的那一列！
        :param c_out:  输出数据维度大小 咱们的任务都是预测单变量， 所以维度为1
        :param seq_len: 窗口大小 ，window_size 大小
        :param label_len: Informer 解码器的起始 token 长度, decoder中 输入的没有掩码部分序列长度
        :param out_len: 预测长度
        :param factor: Probesparse attn因子（默认为5）
        :param d_model: 模型维度
        :param n_heads: 多头注意力头数
        :param e_layers: 编码器层数 默认 2 层
        :param d_layers: 解码器层数 默认 1 层
        :param d_ff: 模型中全连接网络（FCN）的维度，默认值为512
        :param dropout: dropout概率
        :param attn: 编码器中使用的注意事项（默认为prob）。这可以设置为prob（informer）、full（transformer） ，默认为"prob"论文的主要改进点，提出的注意力机制
        :param embed: 时间特征编码（默认为timeF）。这可以设置为时间F、固定、学习
        :param freq: 默认为  小时数据（提供的数据都是小时数居）
        :param activation: 激活函数（默认为gelu）
        :param output_attention:
        :param distil:
        :param mix:
        :param device:
        '''
        # print('seq_len:', seq_len, '  label_len:',label_len,  ' out_len:', out_len)
        # seq_len: 96   label_len: 48   out_len: 1
        # print('c_out:', c_out)
        # c_out: 1
        # 调用父类的初始化方法
        super(InformerBiLSTMModel, self).__init__()
        # 预测长度
        self.pred_len = out_len
        # 注意力机制的选择
        self.attn = attn
        # 是否输出注意力权重  默认不输出
        self.output_attention = output_attention

        # Encoding  # 编码器嵌入层
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # # 解码器嵌入层
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Attention  # 选择注意力机制
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder  # 初始化编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)  # 根据e_layers参数构建多个编码器层
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            # 层归一化
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Decoder  # 初始化解码器
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                # 根据d_layers参数构建多个解码器层
                for l in range(d_layers)
            ],
            # 层归一化
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)

        # BiLSTM参数
        self.num_layers = len(hidden_layer_sizes)  # bilstm层数
        self.bilstm_layers = nn.ModuleList()  # 用于保存BiLSTM层的列表
        # 定义第一层BiLSTM
        self.bilstm_layers.append(nn.LSTM(enc_in, hidden_layer_sizes[0], batch_first=True, bidirectional=True))
        # 定义后续的BiLSTM层
        for i in range(1, self.num_layers):
            self.bilstm_layers.append(
                nn.LSTM(hidden_layer_sizes[i - 1] * 2, hidden_layer_sizes[i], batch_first=True, bidirectional=True))

        # 最终的投影层，将解码器的输出映射到目标输出
        # d_model=128   c_out: 1
        self.projection = nn.Linear(d_model +hidden_layer_sizes[-1]* 2, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        batch_size = x_enc.size(0)
        # print('x_enc.size:', x_enc.size())  # x_enc.size: torch.Size([64, 96, 14])
        # print('x_mark_enc.size:', x_mark_enc.size())  # x_mark_enc.size: torch.Size([64, 96 5])
        # print('x_dec.size:', x_dec.size())   # x_dec.size: torch.Size([64, 49, 14])
        # print('x_mark_dec.size:', x_mark_dec.size())  # x_mark_dec.size: torch.Size([64, 49, 5])

        # 编码器部分
        # 输入数据和时间戳嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print('enc_out.size:', enc_out.size())  # enc_out.size: torch.Size([64, 96, 100])
        # 编码器输出和注意力权重
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print('enc_out.size:', enc_out.size())  # enc_out.size: torch.Size([64, 96, 100])
        # 解码器部分
        # 输入数据和时间戳嵌入
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # print('dec_out.size:', dec_out.size()) # dec_out.size: torch.Size([64, 49, 100])
        # 解码器输出
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # print('dec_out.size:', dec_out.size()) # dec_out.size: torch.Size([64, 49, 100])

        # 送入 BiLSTM 模型
        # 改变输入形状，适应网络输入[batch, seq_length, dim]
        bilstm_out = x_enc  # torch.Size([64, 96, 14])
        hidden = []
        for bilstm in self.bilstm_layers:
            bilstm_out, hidden = bilstm(bilstm_out)  ## 进行一次BiGRU层的前向传播
        # print(bigru_out.size())  # torch.Size([64, 96, 256])

        bilstm_predict = bilstm_out[:, -self.pred_len:, :] # torch.Size([64, 1, 256]
        # print(bigru_gatt_predict.size())  # torch.Size([64, 1, 256])

        # 模型堆叠融合
        combined_features = torch.cat((dec_out[:, -self.pred_len:, :], bilstm_predict), dim=2)  # torch.Size([64, 1, 100 + 256]
        # print(combined_features.size())  # torch.Size([64, 1, 356])

        # 投影到最终输出
        dec_out = self.projection(combined_features)
        # print('dec_out.size:', dec_out.size()) # dec_out.size: torch.Size([64, 1, 1])
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)

        # 根据是否需要输出注意力权重，返回相应的结果
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B, L, D]





