import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import os

class CONV(nn.Module):
    def __init__(self, hidden_activation=F.relu, output_activation=None,hidden_size=500):
        super(CONV, self).__init__()
        
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.conv4 = nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        
        self.hidden = nn.Linear(256 * 8 * 8 + 10, hidden_size)
        self.output = nn.Linear(hidden_size, 2)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation


    def forward(self,image, x_feat):
    
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0),-1)
        x = torch.cat((x,x_feat),dim=1)
        
        x = self.hidden_activation(self.hidden(x))
        x = self.output(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x



val_data = []   
iv_data1 = torch.load("datas/actions_2.pt")
iv_data2 = torch.load("datas/geoms_2.pt")

    
iv_data1 = iv_data1.float()
iv_data2 = iv_data2.float()
iv_data1 = iv_data1.view(-1,1)


iv_data = torch.cat((iv_data1,iv_data2),dim=1)
ov_data = torch.load("datas/positions_2.pt")  

val_data.append((iv_data,ov_data))


im_val = torch.load("datas/imgs_2.pt")

im_val = im_val.float()

x_val = iv_data
y_val = ov_data

model = CONV(hidden_activation=F.sigmoid,output_activation=F.sigmoid,hidden_size=500)
model.load_state_dict(torch.load("conv_model.pth"))
criterion = nn.MSELoss()

with torch.no_grad():
    val_outputs = model(im_val,x_val)
    final_val_loss = criterion(val_outputs, y_val)

print(f'Test Data Loss CNN for Position Estimation: {final_val_loss.item():.8f}')
















