import torch
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
      
        self.conv1 = nn.Conv2d(1,64,3,padding=1)        
        self.conv2 = nn.Conv2d(64,64,3,padding=1)      
        self.pool1 = nn.MaxPool2d(3, 1, padding=1)                 
        self.bn1 = nn.BatchNorm2d(64)                  
        self.relu1 = nn.ReLU()                          
        
        self.conv3 = nn.Conv2d(64,128,3,padding=1)      
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)   
        self.pool2 = nn.MaxPool2d(3, 1, padding=1)      
        self.bn2 = nn.BatchNorm2d(128)                 
        self.relu2 = nn.ReLU()                        

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)    
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)  
        self.conv7 = nn.Conv2d(128, 128, 3,padding=1)   
        self.pool3 = nn.MaxPool2d(3, 1, padding=1)     
        self.bn3 = nn.BatchNorm2d(128)                 
        self.relu3 = nn.ReLU()                        

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)   
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1) 
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1) 
        self.pool4 = nn.MaxPool2d(3, 1, padding=1)    
        self.bn4 = nn.BatchNorm2d(256)                 
        self.relu4 = nn.ReLU()                       

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1) 
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1) 
        self.pool5 = nn.MaxPool2d(3, 1, padding=1)      
        self.bn5 = nn.BatchNorm2d(512)                 
        self.relu5 = nn.ReLU()                         

        self.fc14 = nn.Linear(512*4*4,1024)             
        self.drop1 = nn.Dropout2d()                    
        self.fc15 = nn.Linear(1024,1024)          
        self.drop2 = nn.Dropout2d()                 
        self.fc16 = nn.Linear(1024,10)              

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        
        x = x.view(-1,512*4*4)
        
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x
