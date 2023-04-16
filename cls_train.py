import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from data_process import *
from utils import *
from models import MLP,SVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

is_ml = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def weights_init(m):
    '''模型参数初始化'''
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias.data, 0)


# 读取数据
df_cls = csv2df('weather_.csv')
x, y = get_cls_data(df_cls)

# 数据归一化
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# 划分训练和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True)


if is_ml:
    # 模型
    svm = SVM()
    svc = SVC()
    random_forest = RandomForestClassifier()
    decision_tree = DecisionTreeClassifier()
    
    svm.fit(x_train, y_train)
    svc.fit(X=x_train, y=y_train)
    random_forest.fit(X=x_train, y=y_train)
    decision_tree.fit(X=x_train, y=y_train)

    print("The accuracy of SVM: ", svc.score(x_test, y_test))
    print("The accuracy of DecisionTree: ",
          decision_tree.score(x_test, y_test))
    print("The accuracy of RandomForest: ",
          random_forest.score(x_test, y_test))

else:
    # 超参数
    lr = 5e-3
    epochs = 100
    batch_size = 1024
    # 转化为tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    dataset_train = TensorDataset(x_train, y_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    # 模型
    model = MLP(x_train.shape[-1], 6).to(device)
    model.apply(weights_init)
    # 优化器及损失函数
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    # 训练
    for epoch in range(epochs):
        train_one_epoch_cls(model,optimizer,dataloader_train,device,epoch)
        # for i, (x, y) in enumerate(dataloader_train):
        #     correct = 0
        #     optimizer.zero_grad()
        #     x = x.to(device)
        #     y = (y - 1).to(device)
        #     y_pred = model(x)
        #     loss = criterion(y_pred, y)
        #     loss.backward()
        #     optimizer.step()

        #     predicted = torch.max(y_pred.data, 1)[1]
        #     correct += (predicted == y).sum()

        #     print('Epoch :{}[{}/{}({:.0f}%)]\t Loss:{:.6f}\t Accuracy:{:.3f}'.format(epoch, i * len(x), len(
        #         dataloader_train.dataset), 100.*i / len(dataloader_train), loss.data.item(), float(correct)/float(batch_size)))

