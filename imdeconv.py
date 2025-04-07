import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset,DataLoader
import time
import matplotlib.pyplot as plt
import os


class Data_set(Dataset):
    def __init__(self,imgs,xfeats,outputs):
        self.imgs = imgs
        self.xfeats = xfeats
        self.outputs = outputs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,idx):
        img = self.imgs[idx]
        xfeat = self.xfeats[idx]
        output = self.outputs[idx]
        return img,xfeat,output


class DECONV(nn.Module):
    def __init__(self, hidden_activation=F.relu, output_activation=None,hidden_size=500):
        super(DECONV, self).__init__()
        
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.conv4 = nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        
        self.hidden = nn.Linear(256 * 8 * 8 + 10, hidden_size)
        self.outputxy = nn.Linear(hidden_size, 2)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.outputimg = nn.Linear(hidden_size, 256 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1)
        self.deconv2 = nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1)
        self.deconv3 = nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1)
        self.deconv4 = nn.ConvTranspose2d(32,3,kernel_size=4,stride=2,padding=1)
        #self.resize = nn.Upsample(size = (32,32),mode="bilinear",align_corners=True)
        #self.xd = nn.Linear(2,256*8*8)

    def forward(self,image, x_feat):
    
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0),-1)
        x = torch.cat((x,x_feat),dim=1)
        
        x = self.hidden_activation(self.hidden(x))

        xy = self.outputxy(x)

        if self.output_activation:
            xy = self.output_activation(xy)

        
        xim = self.outputimg(x)
        xim = xim.view(x.size(0),256,8,8)
        #print(xim.shape)
        xim = F.relu(self.deconv1(xim))
        xim = F.relu(self.deconv2(xim))
        xim = F.relu(self.deconv3(xim))
        xim = torch.sigmoid(self.deconv4(xim))
        #xim = self.resize(xim)
        return xy,xim

def show_image(ref_img,gen_img, title_ref="Ref Image",title_gen = "Generated Image"):
    fig,axes = plt.subplots(1,2,figsize=(12,6))

    ref_img = ref_img.detach().cpu().numpy().transpose(1,2,0)
    
    axes[0].imshow(ref_img)
    axes[0].set_title(title_ref)
    axes[0].axis("off")

    gen_img = gen_img.detach().cpu().numpy().transpose(1,2,0)
    axes[1].imshow(gen_img)
    axes[1].set_title(title_gen)
    axes[1].axis("off")

    plt.show()


def float_to_uint8(img):
    img = img.detach().cpu()
    img = img.clamp(0,1)
    uint8_img = (img * 255).byte()
    return uint8_img




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

im_train_u8 = torch.load("datas/imgs_0.pt")
im_val_u8 = torch.load("datas/imgs_1.pt")

im_train = im_train_u8.float()
im_val = im_val_u8.float()

im_train = im_train / 255.0
im_val = im_val / 255.0
im_train = im_train[:200]

x_train = it_data[:200]
y_train = ot_data[:200]
x_val = iv_data
y_val = ov_data

trainset = Data_set(im_train,x_train,y_train)
valset = Data_set(im_val,x_val,y_val)

trainloader = DataLoader(trainset, batch_size = 20,shuffle = True)
valloader = DataLoader(valset, batch_size = 20,shuffle = False)

model = DECONV(hidden_activation=F.sigmoid,output_activation=F.sigmoid,hidden_size=500)
criterion = nn.MSELoss()

initial_lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=initial_lr)

#scheduler = StepLR(optimizer, step_size = 250,gamma = 0.1)


Plot_Loss = []
Plot_Val_Loss = []
Plot_Epoch = []

# Training loop
num_epochs = 500
t_start = time.time() 
for epoch in range(num_epochs):
    for im_traint,x_traint,y_traint in trainloader:
        optimizer.zero_grad()
        outputs,img_outputs = model(im_traint,x_traint)
        loss_xy = criterion(outputs, y_traint)
        loss_img = criterion(img_outputs,im_traint)
        loss = loss_xy + loss_img
        loss.backward()
        optimizer.step()
        #scheduler.step()
   
    if epoch % 10 == 0:
        with torch.no_grad():
            for im_valt,x_valt,y_valt in valloader:
                val_outputs,val_img_outputs= model(im_valt,x_valt)
                val_loss_xy = criterion(val_outputs, y_valt)
                val_loss_img = criterion(val_img_outputs,im_valt)
                val_loss = val_loss_xy + val_loss_img
                #val_loss = criterion(val_outputs, y_val)
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}, Val Loss: {val_loss.item():.8f}')
        Plot_Loss.append(loss.item())
        Plot_Val_Loss.append(val_loss.item())
        Plot_Epoch.append(epoch)
        

# Final validation loss
with torch.no_grad():
    for f_im_valt,f_x_valt,f_y_valt in valloader:
        f_val_outputs,f_val_loss_img = model(f_im_valt,f_x_valt)
        f_val_loss_xy = criterion(val_outputs, f_y_valt)
        f_val_loss_img = criterion(val_img_outputs,f_im_valt)
        final_val_loss = f_val_loss_xy + f_val_loss_img
        #final_val_loss = criterion(f_val_outputs, f_y_valt)
t_stop = time.time()
print(f'Final Validation Loss: {final_val_loss.item():.8f}')
print(f"Elapsed time seconds: {t_stop-t_start:.4f}")

with torch.no_grad():
    sample_xy,sample_image=model(im_val[:1],x_val[:1])
    generated_image = sample_image.squeeze(0)
    generated_image = float_to_uint8(generated_image)
    generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())
    reference_image = im_val_u8[1].squeeze(0)
    show_image(reference_image,generated_image,title_ref="Reference Image",title_gen="Generated Image")


save_dir = "time_plots"
os.makedirs(save_dir,exist_ok = True)

plt.plot(Plot_Epoch,Plot_Loss, label="Training")
plt.plot(Plot_Epoch,Plot_Val_Loss, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()

file_path = os.path.join(save_dir,"DCNN_Error_Plot.png")
plt.savefig(file_path)


torch.save(model.state_dict(),"deconv_model.pth")















