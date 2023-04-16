import torch.nn.functional as F
from torch import nn
import numpy as np
import torch
import math


class MLP(nn.Module):
    def __init__(self, in_feat, num_cls) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_feat, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.predict = nn.Linear(32, num_cls)
        self.dropout = nn.Dropout(0.25)
        self.norm = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.norm(self.fc2(x)))
        x = self.dropout(self.fc3(x))
        x = F.relu(self.norm(self.fc4(x)))
        x = F.relu(self.fc5(x))
        x = F.softmax(self.predict(x))

        return x


class SVM():
    def __init__(self, max_iter=1, kernel='linear', degree=1):
        self.max_iter = max_iter
        self._kernel = kernel
        self.degree = degree

    # 定义SVM的参数
    def define_arguments(self, X, y):
        self.row, self.col = X.shape  # 特征的行和列
        self.X = X  # 数据
        self.Y = y  # 标签
        self.b = 0.0  # 偏置
        self.a = np.ones(self.row)  # 参数阿尔法
        self.error = [self.cal_error(i) for i in range(self.row)]  # 标签y与真实值的误差
        self.C = 1  # 松弛变量

    # 检查下标为i的划分是否满足kkt
    def kkt_condition(self, i):
        '''基于SVM理论，即f(x)是最优超平面'''
        split = self.f(i) * self.Y[i]  # 原理：yi*(wT*xi + b),即是否能被超平面划分

        if self.a[i] == 0:
            return split >= 1
        elif 0 < self.a[i] < self.C:
            return split == 1
        else:
            return split <= 1

    # f(x),表示对x的预测值

    def f(self, x):
        y_pred = 0  # 模型预测值
        for i in range(self.row):
            # 原理:f(x) = 累加 ai*yi*xiT*xi + b
            y_pred += self.a[i] * self.Y[i] * \
                self.kernel(self.X[x], self.X[i])  # 利用核函数对数据x进行变换
        return y_pred + self.b

    # 核函数的设计，
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return sum([x1[i] * x2[i] for i in range(self.col)])
        elif self._kernel == 'poly':
            return (sum([x1[i] * x2[i] for i in range(self.col)]) + 1) ** self.degree

        return 0

    # E（x）为g(x)对输入x的预测值和y的差，这里的x表示下标
    def cal_error(self, x):
        return self.f(x) - self.Y[x]

    #更新参数a，选|error1 - error2|最大的来更新，因为误差越大，更新变换越大
    def updata_a(self):
        for i in range(self.row):
            if self.kkt_condition(i):
                continue  # 满足条件则不变

            error1 = self.error[i]  # 固定a1，计算器误差
            j = 0  # a2的下标

            # 不满足条件则优化a1和a2,选|error1 - error2|最大的来更新
            # error1>0,则error2越小，相减绝对值越大，小于0同理
            if error1 >= 0:
                error_temp = self.error[j]  # 记录当前最小值
                for k in range(self.row):
                    if self.error[k] < error_temp:
                        j = k
                        error_temp = self.error[k]
            else:
                error_temp = self.error[j]  # 记录当前最小值
                for k in range(self.row):
                    if self.error[k] > error_temp:
                        j = k
                        error_temp = self.error[k]

            return i, j

    def sequential_minimal_optimization(self):
        index1, index2 = self.updata_a()  # 固定i1，从而更新i2

        # 计算上下界，且新的a1和a2必须L~H内
        if self.Y[index1] == self.Y[index2]:
            L = max(0, self.a[index1] + self.a[index2] - self.C)
            H = min(self.C, self.a[index1] + self.a[index2])
        else:
            L = max(0, self.a[index2] - self.a[index1])
            H = min(self.C, self.C + self.a[index2] - self.a[index1])

        error1 = self.error[index1]
        error2 = self.error[index2]

        # n = K11+K22-2K12，这里Kij表示核函数
        n = self.kernel(self.X[index1], self.X[index1]) + self.kernel(self.X[index2],
                                                                      self.X[index2]) - 2 * self.kernel(
            self.X[index1], self.X[index2])

        # 如果n小于等于0，表示满足条件，不需要更新
        if n <= 0:
            return

        # 该部分为更新参数，参考博客
        a2_new_unc = self.a[index2] + self.Y[index2] * (error1 - error2) / n
        a2_new = self.check_range(a2_new_unc, L, H)

        a1_new = self.a[index1] + self.Y[index1] * \
            self.Y[index2] * (self.a[index2] - a2_new)

        b1_new = -error1 - self.Y[index1] * self.kernel(self.X[index1], self.X[index1]) * (a1_new - self.a[index1]) - self.Y[
            index2] * self.kernel(self.X[index2], self.X[index1]) * (a2_new - self.a[index2]) + self.b
        b2_new = -error2 - self.Y[index1] * self.kernel(self.X[index1], self.X[index2]) * (a1_new - self.a[index1]) - self.Y[
            index2] * self.kernel(self.X[index2], self.X[index2]) * (a2_new - self.a[index2]) + self.b

        # 判断a1_new和a2_new是否为0或C
        if 0 < a1_new < self.C:
            b_new = b1_new
        elif 0 < a2_new < self.C:
            b_new = b2_new
        else:
            b_new = (b1_new + b2_new) / 2  # a1_new 和a2_new是0或C

        # 更新参数
        self.a[index1] = a1_new
        self.a[index2] = a2_new
        self.b = b_new

        self.error[index1] = self.cal_error(index1)
        self.error[index2] = self.cal_error(index2)

    # 比较是a是否再范围内
    def check_range(self, a, L, H):
        # 若在上下界范围外，则归到上界或下界
        if a > H:
            return H
        elif a < L:
            return L
        else:
            return a

    def fit(self, X, y):
        self.define_arguments(X, y)

        # 根据最大迭代次数运用smo更新参数
        for t in range(self.max_iter):
            self.sequential_minimal_optimization()

    def predict(self, x):
        y_pred = self.b
        for i in range(self.row):
            y_pred += self.a[i] * self.Y[i] * self.kernel(x, self.X[i])

        return 1 if y_pred > 0 else 0

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)

    def _weight(self):
        # linear model
        yx = self.Y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.a)
        return self.w


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class Transformer(nn.Module):
    def __init__(self,input_dim,d_model,output_dim,seq_len) -> None:
        super(Transformer,self).__init__()
        self.output_dim = output_dim
        self.input_emb = nn.Linear(input_dim,d_model)
        self.output_emb = nn.Linear(output_dim,d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer,6)
        self.decoder = nn.TransformerDecoder(self.decoder_layer,6)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(*[
            nn.Linear(seq_len*d_model,d_model),
            nn.Linear(d_model,output_dim)
        ])
        
    def forward(self,x):
        # y = x[:,:,-1]
        x = self.input_emb(x)
        x = self.pos_emb(x)
        x = self.encoder(x)
        # y = self.output_emb(y)
        # out = self.decoder(y,x)
        out = self.flatten(x)
        out = self.mlp(out)
        
        return out


class RNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,num_layer,device="cpu",is_gru=True) -> None:
        super(RNN,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.is_gru = is_gru
        self.device = device
        self.lstm = nn.LSTM(self.input_dim,self.hidden_dim,self.num_layer,batch_first=True)
        self.gru = nn.GRU(self.input_dim,self.hidden_dim,self.num_layer,batch_first=True)
        self.fc = nn.Linear(self.hidden_dim,self.output_dim)
        
    def forward(self,x):
        batch_size,seq_len = x.shape[0],x.shape[1]
        h_0 = torch.randn(self.num_layer,batch_size,self.hidden_dim).to(self.device)
        c_0 = torch.randn(self.num_layer,batch_size,self.hidden_dim).to(self.device)
            
        if self.is_gru:
            output,_ = self.gru(x,(h_0,c_0))
        else:
            output,_ = self.lstm(x,(h_0,c_0))
        
        pred = self.fc(output)
        pred = pred[:,-1,:]
        
        return pred
        