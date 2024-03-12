import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#导入数据
df=pd.read_excel("2024年2月美赛代码/Wimbledon_featured_matches_editted.xlsx",sheet_name="Sheet1")
#print(df)
#print(df.shape)
length=df.shape[0]
df_result=pd.read_excel("2024年2月美赛代码/Wimbledon_featured_matches_editted.xlsx",sheet_name="Sheet2")
df_result=df_result.iloc[:,0]
X=df.to_numpy()
Y=df_result.to_numpy()
Y=np.array([Y.T for i in range(11)]).T
#print(Y.shape)
#超参数
BATCH_SIZE=12
INPUT_SIZE=1
HIDDEN_SIZE=32
NUM_LAYERS=16
LR=0.02
EPOCHS=length
#print(X,Y)
#RNN
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )
        self.out=nn.Linear(32,1)
    def forward(self,x,h_state):
        r_out,h_state=self.rnn(x,h_state)
        outs=[]
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:,time,:]))
        return torch.stack(outs,dim=1),h_state

model=RNN()
Loss_func=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
h_state=None
#print(X[0:100,10])
#循环训练
for i in range(0,EPOCHS,12):
    if i+BATCH_SIZE<=EPOCHS:
        x=torch.unsqueeze(torch.from_numpy(X[i:i+BATCH_SIZE]),2).float()
        y=torch.unsqueeze(torch.from_numpy(Y[i:i+BATCH_SIZE]),2).float()
    else:
        x=torch.unsqueeze(torch.from_numpy(X[i:EPOCHS]),2).float()
        y=torch.unsqueeze(torch.from_numpy(Y[i:EPOCHS]),2).float()
    #print(x[0][10][0])
    #print(x.shape)
    #print(y[0][0][0])
    #print(y.shape)
    prediction,h_state=model(x,h_state)
    h_state=h_state.data
    loss=Loss_func(prediction,y)
    if i%144==0:
        print("Epoch:",i,"Loss:",loss.data.numpy(),";Accuracy:",)
    #print(prediction[0][0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(prediction.shape)
print(y.shape)