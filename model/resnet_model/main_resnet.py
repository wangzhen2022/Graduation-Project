import  torch
from  torch import nn, optim
import pandas
import numpy as np
import torch.utils.data as data
from torch.utils.data import TensorDataset
import torchmetrics
from model import ResNet18
import matplotlib.pyplot as plt

def main():

    #设置使用cpu或gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x_train = pandas.read_csv('../../train_test_datasets/X_train.csv', header=None)
    y_train = pandas.read_csv('../../train_test_datasets/Y_train.csv', header=None)
    x_test = pandas.read_csv('../../train_test_datasets/X_test.csv', header=None)
    y_test = pandas.read_csv('../../train_test_datasets/Y_test.csv', header=None)


    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    #改变数据维度
    x_train = x_train.reshape(x_train.shape[0],1,4,4)
    x_test = x_test.reshape(x_test.shape[0], 1,4,4)

    #数据融合
    train_set = TensorDataset(torch.Tensor(x_train).type(torch.float32),torch.Tensor(y_train).type(torch.long))
    test_set = TensorDataset(torch.Tensor(x_test).type(torch.float32),torch.Tensor(y_test).type(torch.long))

    #创建数据加载器
    train_loader=data.DataLoader(train_set,batch_size=100,shuffle=True)
    test_loader=data.DataLoader(test_set,batch_size=100,shuffle=True)

    # 定义模型-ResNet
    model = ResNet18().to(device)

    
    #模型训练和测试
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    loss_count = []
    
    for epoch in range(10):
        running_loss = 0.0
        for i, (input_data,labels)  in enumerate(train_loader):
            input_data = input_data.to(device, torch.float)
            labels = labels.to(device, torch.long)

            outputs = model(input_data)
            loss = criterion(outputs, labels.squeeze(1).long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            running_loss += loss.item()   
            
            if i % 200 == 199:
                print('[%d, %5d] loss: %0.3f' % (epoch + 1, i + 1, running_loss / 200))
                loss_count.append(running_loss/200)
                running_loss = 0.0

    plt.figure('PyTorch_ResNet_Loss')
    plt.plot(loss_count,label='Loss')
    plt.legend()
    plt.savefig("../../picture/resnet_loss.jpg") 
    plt.show()
    
              
    
    acc = torchmetrics.Accuracy().to(device)
    recall = torchmetrics.Recall(average='none', num_classes=10).to(device)
    precision = torchmetrics.Precision(average='none', num_classes=10).to(device)
    f1 = torchmetrics.F1(average='none', num_classes=10).to(device)

    print("test_action")

    x = torch.tensor([[0]]).to(device)
    y = torch.tensor([[0]]).to(device)
    with torch.no_grad():
        for samples, labels1 in test_loader:
            samples = samples.to(device, torch.float)
            labels1 = labels1.to(device, torch.long)
                    
            outputs1 = model(samples)
           
            _, pred1 = torch.max(outputs1.data,1)

            pred1 = pred1.reshape(-1,1)

            x = torch.cat((x,pred1),0).to(device)
            y = torch.cat((y,labels1),0).to(device)
    
       
    print("text_acc:",acc(x,y))
    print("test_precision:",precision(x,y))
    print("test_recall:",recall(x,y))
    print("test_f1:",f1(x,y))


if __name__ == '__main__':
    main()