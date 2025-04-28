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
iv_data1 = torch.load("datas/actions_1.pt")
iv_data2 = torch.load("datas/geoms_1.pt")

    
iv_data1 = iv_data1.float()
iv_data2 = iv_data2.float()
iv_data1 = iv_data1.view(-1,1)


iv_data = torch.cat((iv_data1,iv_data2),dim=1)
ov_data = torch.load("datas/positions_1.pt")  


train_data = []
it_data1 = torch.load("datas/actions_0.pt")
it_data2 = torch.load("datas/geoms_0.pt")

it_data1 = it_data1.float()
it_data2 = it_data2.float()
it_data1 = it_data1.view(-1,1)

it_data = torch.cat((it_data1,it_data2),dim=1)
ot_data = torch.load("datas/positions_0.pt")


train_data.append((it_data,ot_data))

val_data.append((iv_data,ov_data))

im_train = torch.load("datas/imgs_0.pt")
im_val = torch.load("datas/imgs_1.pt")

im_train = im_train.float()
im_val = im_val.float()

x_train = it_data
y_train = ot_data
x_val = iv_data
y_val = ov_data

model = CONV(hidden_activation=F.sigmoid,output_activation=F.sigmoid,hidden_size=500)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
Plot_Loss = []
Plot_Val_Loss = []
Plot_Epoch = []
# Training loop
num_epochs = 100
t_start = time.time() 
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(im_train,x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
   
    if epoch % 10 == 0:
        with torch.no_grad():
            val_outputs = model(im_val,x_val)
            val_loss = criterion(val_outputs, y_val)
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}, Val Loss: {val_loss.item():.8f}')
        Plot_Loss.append(loss.item())
        Plot_Val_Loss.append(val_loss.item())
        Plot_Epoch.append(epoch)

# Final validation loss
with torch.no_grad():
    val_outputs = model(im_val,x_val)
    final_val_loss = criterion(val_outputs, y_val)
t_stop = time.time()
print(f'Final Validation Loss: {final_val_loss.item():.8f}')
print(f"Elapsed time seconds: {t_stop-t_start:.4f}")

save_dir = "/time_plots"
os.makedirs(save_dir,exist_ok = True)

plt.plot(Plot_Epoch,Plot_Loss, label="Training")
plt.plot(Plot_Epoch,Plot_Val_Loss, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()

file_path = os.path.join(save_dir,"CNN_Error_Plot.png")
plt.savefig(file_path)


torch.save(model.state_dict(),"conv_model.pth")















