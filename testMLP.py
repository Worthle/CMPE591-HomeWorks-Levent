import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import os

class MLP(nn.Module):
    def __init__(self, hidden_activation=F.relu, output_activation=None,hidden_size=500):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(10, hidden_size)
        self.output = nn.Linear(hidden_size, 2)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def forward(self, x):
        x = self.hidden_activation(self.hidden(x))
        x = self.output(x)
        if self.output_activation:
        	x = self.output_activation(x)
        return x




val_data = []	
iv_data1 = torch.load("datas/actions_2.pt")
iv_data2 = torch.load("datas/geoms_2.pt")
ov_data = torch.load("datas/positions_2.pt")	
	
iv_data1 = iv_data1.float()
iv_data2 = iv_data2.float()
iv_data1 = iv_data1.view(-1,1)


iv_data = torch.cat((iv_data1,iv_data2),dim=1)


val_data.append((iv_data,ov_data))


x_val = iv_data
y_val = ov_data

model = MLP(hidden_activation=F.relu,output_activation=F.tanh,hidden_size=500)
model.load_state_dict(torch.load("mlp_model.pth"))
criterion = nn.MSELoss()


with torch.no_grad():
    val_outputs = model(x_val)
    final_val_loss = criterion(val_outputs, y_val)

print(f'Test Data Loss MLP for Position Estimation: {final_val_loss.item():.8f}')
















