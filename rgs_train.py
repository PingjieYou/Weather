import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from data_process import *
from utils import *
import dataset
from models import MLP, SVM, Transformer,RNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df_rgs = csv2df('chongqing_.csv')

## 超参数
lr = 1e-4
epochs = 20
batch_size = 32
device = "cuda:0" if torch.cuda.is_available() else "cpu"

## 数据
weather_dataset = dataset.Weather(df=df_rgs, seq_len=7)
weather_dataloader = DataLoader(
    dataset=weather_dataset, batch_size=batch_size, shuffle=True)

## 模型及优化器损失函数
rnn =RNN(2,32,1,4,device,False).to(device)
transformer = Transformer(2, 512, 1, 7).to(device)
optimizer_r = torch.optim.Adam(lr=1e-3, params=rnn.parameters())
optimizer_t = torch.optim.Adam(lr=1e-5, params=transformer.parameters())
criterion = nn.MSELoss()

loss_list_r = []
loss_list_t = []
for epoch in range(epochs):
    loss_r = utils.train_one_epoch_rgs(rnn,optimizer_r,weather_dataloader,device,epoch)
    loss_t = utils.train_one_epoch_rgs(transformer,optimizer_t,weather_dataloader,device,epoch)
    loss_list_r.extend(loss_r)
    loss_list_t.extend(loss_t)

plt.plot(range(1,len(loss_list_r)+1),loss_list_r,label='RNN',color='r')
plt.plot(range(1,len(loss_list_t)+1),loss_list_t,label='Transformer',color='b')
plt.xlabel(xlabel="times")
plt.ylabel(ylabel="loss")
plt.legend()
plt.show()

y_pred_list_r = []
y_pred_list_t = []
y_list = []
dataloader = DataLoader(weather_dataset,batch_size=128,shuffle=False)
for x,y in dataloader:
    x = x.to(device)
    y = y.to(device)
    
    pred_r = rnn(x)[:,0].detach().cpu().data.numpy()
    pred_t = transformer(x)[:,0].detach().cpu().data.numpy()
    y_pred_list_r.extend(pred_r) 
    y_pred_list_t.extend(pred_t)    
    y_list.extend(y.detach().cpu().data.numpy())

    
plt.plot(y_pred_list_r,label='rnn',color='r')
plt.plot(y_pred_list_t,label='transformer',color='b')
plt.plot(y_list,label='true',color='green')
plt.xlabel(xlabel="date")
plt.ylabel(ylabel="temperature")
plt.legend()
plt.show()
plt.show()