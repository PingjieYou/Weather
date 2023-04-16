import torch
import utils
import data_process
from torch.utils.data import Dataset,DataLoader

class Weather(Dataset):
    '''重庆天气回归数据集'''
    def __init__(self,df,seq_len=7) -> None:
        super(Weather,self).__init__()
        temp = data_process.get_rgs_data(df)
        self.x = torch.tensor([temp[i:i+seq_len] for i in range(len(temp)-seq_len)],dtype=torch.float32)
        self.y = torch.tensor([temp[i] for i in range(seq_len,len(temp))],dtype=torch.float32)[:,-1:]

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)
