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
iv_data1 = torch.load("datas/actions_1.pt")
iv_data2 = torch.load("datas/geoms_1.pt")
ov_data = torch.load("datas/positions_1.pt")	
	
iv_data1 = iv_data1.float()
iv_data2 = iv_data2.float()
iv_data1 = iv_data1.view(-1,1)


iv_data = torch.cat((iv_data1,iv_data2),dim=1)

train_data = []


it_data1 = torch.load("datas/actions_0.pt")
it_data2 = torch.load("datas/geoms_0.pt")

it_data1 = it_data1.float()
it_data2 = it_data2.float()
it_data1 = it_data1.view(-1,1)

it_data = torch.cat((it_data1,it_data2),dim=1)


ot_data = torch.load("datas/positions_0.pt")

#ot_data = torch.cat((ot_data1),dim=0)

#iv_data = iv_data.view(-1,1)

train_data.append((it_data,ot_data))

val_data.append((iv_data,ov_data))




x_train = it_data
y_train = ot_data
x_val = iv_data
y_val = ov_data

model = MLP(hidden_activation=F.relu,output_activation=F.tanh,hidden_size=500)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

Plot_Loss = []
Plot_Val_Loss = []
Plot_Epoch = []
# Training loop
num_epochs = 1000
t_start = time.time() 
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
   
    if epoch % 10 == 0:
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}, Val Loss: {val_loss.item():.8f}')
        Plot_Loss.append(loss.item())
        Plot_Val_Loss.append(val_loss.item())
        Plot_Epoch.append(epoch)


# Final validation loss
with torch.no_grad():
    val_outputs = model(x_val)
    final_val_loss = criterion(val_outputs, y_val)
t_stop = time.time()
print(f'Final Validation Loss: {final_val_loss.item():.8f}')
print(f"Elapsed time seconds: {t_stop-t_start:.4f}")


save_dir = "/home/levubuntu/cmpe591.github.io/src/time_plots"
os.makedirs(save_dir,exist_ok = True)

plt.plot(Plot_Epoch,Plot_Loss, label="Training")
plt.plot(Plot_Epoch,Plot_Val_Loss, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()

file_path = os.path.join(save_dir,"MLP_Error_Plot.png")
plt.savefig(file_path)

torch.save(model.state_dict(),"/home/levubuntu/cmpe591.github.io/src/mlp_model.pth")
















