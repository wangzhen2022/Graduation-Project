from regex import P
import torch.nn as nn
import torch.nn.functional as F

class lstm(nn.Module):
    def __init__(self,input_size=4,hidden_size=512,output_size=10,num_layer=6):
        super(lstm,self).__init__()

        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer,batch_first=True)
        self.layer2 = nn.Linear(2048,hidden_size)
        self.layer3 = nn.Linear(hidden_size,output_size)
        
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        # x = x.contiguous().view(s,b*h)
        x = x.contiguous().view(-1,b*h)
        print(x.size())
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x