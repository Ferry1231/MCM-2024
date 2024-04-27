#导入相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchmetrics
import xlsxwriter
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from SALib.sample import saltelli
from SALib.analyze import sobol

#导入数据
df=pd.read_excel("2024年2月美赛代码/Wimbledon_featured_matches_editted.xlsx",sheet_name="Sheet1")
length=df.shape[0]
df_result=pd.read_excel("2024年2月美赛代码/Wimbledon_featured_matches_editted.xlsx",sheet_name="Sheet2")
df_result=df_result.iloc[:,0]
X=df[:300].to_numpy()
Y_0=df_result[:300].to_numpy()
for i in range(Y_0.shape[0]):
    Y_0[i]=Y_0[i]+13
Y=[[0 for i in range(27)] for j in range(Y_0.shape[0])]
for i in range(Y_0.shape[0]):
    Y[i][Y_0[i]-1]=1

#超参数
BATCH_SIZE=64
EPOCHES=300
KERNEL_SIZE=1
STRIDE=1
LR=0.001

#搭建模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv1d(1,10,KERNEL_SIZE,stride=STRIDE)
        self.max_pool1=nn.MaxPool1d(KERNEL_SIZE,stride=STRIDE)
        self.conv2=nn.Conv1d(10,20,KERNEL_SIZE,stride=STRIDE)
        self.max_pool2=nn.MaxPool1d(KERNEL_SIZE,stride=STRIDE)
        self.conv3=nn.Conv1d(20,40,KERNEL_SIZE,stride=STRIDE)
        self.linear1=nn.Linear(40*X.shape[1],120)
        self.linear2=nn.Linear(120,84)
        self.linear3=nn.Linear(84,27)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.max_pool1(x)
        x=F.relu(self.conv2(x))
        x=self.max_pool2(x)
        x=F.relu(self.conv3(x))
        x=x.view(-1,40*X.shape[1])  
        x=F.relu(self.linear1(x))
        x=F.relu(self.linear2(x))
        x=self.linear3(x)
        x=self.softmax(x)
        return x

#模型&损失函数
model=CNN()
Loss_func=nn.MSELoss(size_average=None, reduce=None, reduction='sum')
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
accuracy=torchmetrics.Accuracy(task="multiclass",num_classes=27)
f1=torchmetrics.F1Score(task="multiclass",num_classes=27)
recall=torchmetrics.Recall(task="multiclass",num_classes=27)
precision=torchmetrics.Precision(task="multiclass",num_classes=27)

#设备挂载
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device=torch.device("cpu")
model.to(device)
Loss_func.to(device)

#数据处理
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
Y_train=torch.tensor(Y_train,dtype=torch.float32)
Y_test=torch.tensor(Y_test,dtype=torch.float32)

X_train=X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test=X_test.reshape(X_test.shape[0],1,X_test.shape[1])
Y_train=Y_train.reshape(Y_train.shape[0],1,Y_train.shape[1])
Y_test=Y_test.reshape(Y_test.shape[0],1,Y_test.shape[1])

#循环训练&tensorboard绘制训练曲线
writer_vi = SummaryWriter(log_dir="2024年2月美赛代码/logs")
for epoch in tqdm(range(100)):
    running_loss=0#误差重置
    accuracy.reset()
    f1.reset()
    recall.reset()
    precision.reset()
    for i,input in enumerate(X_train,0):
        label=Y_train[i]
        label=label.to(device)
        optimizer.zero_grad()
        input=input.to(device)
        output=model(input)
        output=output.to(device)
        loss=Loss_func(output,label)
        accuracy.update(output,label)
        f1.update(output,label)
        recall.update(output,label)
        precision.update(output,label)
        loss.backward()
        optimizer.step()
        writer_vi.add_scalar("train_loss",loss.item(),epoch)
        running_loss+=loss.item()
        if i%99==0:
            print('[%d,%5d]loss:%0.6f'%(epoch+1,i+1,running_loss/2000))
            running_loss=0
    with torch.no_grad():
        acc=1-Loss_func(model(X_test[0]),Y_test[0])
        writer_vi.add_scalar("test_acc",acc,epoch)         
writer_vi.close()

#模型预测准确率
Accuracy=accuracy.compute()
F1_score=f1.compute()
Recall=recall.compute()
Precision=precision.compute()
print('|Accurary:%0.6f|F1-score:%0.6f|Recall-ratio:%0.6f|Precision:%0.6f|'%(Accuracy,F1_score,Recall,Precision))

#绘图
Pre_test_data_0=[]
Act_test_data_0=[]
Pre_test_data=[]
Act_test_data=[]
Acc=[]
Pre=0
Act=0

X_1=torch.tensor(X,dtype=torch.float32)
X_1=X_1.reshape(X_1.shape[0],1,X_1.shape[1])
count=np.array([i for i in range(Y_test.shape[2])])

for i in range(X_1.shape[0]):
    Pre_test_data_0.append(model(X_1[i]).detach().numpy())
    Act_test_data_0.append(Y[i])
for i in range(X_1.shape[0]):
    Pre_data=np.dot(Pre_test_data_0[i],count)
    Pre=np.argmax(Pre_test_data_0[i])-13
    Act_data=np.dot(Act_test_data_0[i],count)
    Act=np.argmax(Act_test_data_0[i])-13
    Pre_test_data.append(Pre)
    Act_test_data.append(Act)
    Acc.append(1-abs(Pre_data-Act_data)/(Act_data))

x_state=np.array([i for i in range(X_1.shape[0])])
Pre_test_data=np.array(Pre_test_data)
Act_test_data=np.array(Act_test_data)

color_1=np.array([100 for i in range(X_1.shape[0])])
color_2=np.array([200 for i in range(X_1.shape[0])])
size=np.array([1 for i in range(X_1.shape[0])])

plt.plot(x_state,Pre_test_data,'r',label="Predicted")
plt.plot(x_state,Act_test_data,'b',label="Actual")
plt.xlabel("X_State")
plt.ylabel("Advantage Relative")
plt.legend()
plt.grid(True)
plt.title("Wimbledon 2023 Gentlemen's Single")
plt.show()

#计算模型准确率
print("Accuracy:%0.6f"%np.average(Acc))
'''
#CNN参数及其存储
params = model.parameters()
writer = pd.ExcelWriter("2024年2月美赛代码/CNN_Weights.xlsx", engine='xlsxwriter')
for param in params:
    array=param.data.detach().numpy()
    if param.ndim>1:
        array=array.reshape(param.shape[0],param.shape[1])
    elif param.ndim==1:
        array=array.reshape(param.shape[0],1)
    df=pd.DataFrame(array)
    sheetname="torch "+str(param.shape[0])
    if param.ndim>1:
        sheetname += " " + str(param.shape[1])
    df.to_excel(writer,sheet_name=sheetname)
writer.close()
'''

#灵敏度分析
'''problem = {
    'num_vars': Y_0.shape[0], 
    'bounds': [[-2,2],
               [-3,3],
               [-40,40],
               [-1,1],
               [-1,1],
               [-1,1],
               [-1,1],
               [-1,1],
               [-1,1],
               [-1,1],
               [0,3]]}
param_values = torch.tensor(saltelli.sample(problem, 1000))
param_values.reshape(param_values.shape[0],1,param_values.shape[1])
param_pre=model(param_values)
print(param_pre.shape,param_values.shape)
Si = sobol.analyze(problem,param_pre, print_to_console=True)
print('S1:', Si['S1'])
print("x1-x2:", Si['S2'][0, 1])
print("x1-x3:", Si['S2'][0, 2])
print("x2-x3:", Si['S2'][1, 2])
Si.plot()'''
'''
train_test_split|accurary|F1_score
0.1 0.951507(0.7486%) 0.004115
0.15 0.953130(0.9109%) 0.002760
0.2 0.944021(0.0000%) 0.000154
0.25 0.947785(0.3764%) 0.000165
0.3 0.941085(-0.2936%) 0.000353
'''