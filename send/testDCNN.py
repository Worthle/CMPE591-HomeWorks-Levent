import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
#from imdeconv import DECONV,Data_set

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

def show_image(file_name,ref_img,gen_img, title_ref="Ref Image",title_gen = "Generated Image"):
    fig,axes = plt.subplots(1,2,figsize=(12,6))

    ref_img = ref_img.detach().cpu().numpy().transpose(1,2,0)
    
    axes[0].imshow(ref_img)
    axes[0].set_title(title_ref)
    axes[0].axis("off")

    gen_img = gen_img.detach().cpu().numpy().transpose(1,2,0)
    axes[1].imshow(gen_img)
    axes[1].set_title(title_gen)
    axes[1].axis("off")

    plt.savefig(file_name,bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def float_to_uint8(img):
    img = img.detach().cpu()
    img = img.clamp(0,1)
    uint8_img = (img * 255).byte()
    return uint8_img


model = DECONV()

model.load_state_dict(torch.load("deconv_model.pth"))

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
im_val = im_val / 255.0
im_valt = im_val.float()
output_dir = "test_images"
os.makedirs(output_dir, exist_ok=True)


valset = Data_set(im_val,iv_data,ov_data)
valloader = DataLoader(valset, batch_size = 200,shuffle = False)
criterion = nn.MSELoss()
model.eval()

with torch.no_grad():
    for im_valt,x_valt,y_valt in valloader:
        outputs,img_outputs = model(im_valt,x_valt)
        loss_xy = criterion(outputs, y_valt)
        loss_img = criterion(img_outputs,im_valt)
        loss = loss_xy + loss_img
        print(f'Total Image Error Between Generated and Reference: {loss_img.item():.8f}')



"""
for i in range(20):
    for im_valt,x_valt,y_valt in valloader:
        sample_xy, sample_image = model(im_valt, x_valt)
    
        generated_image = sample_image[i].squeeze(0) 

        generated_image = float_to_uint8(generated_image)
        generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())

        reference_image = im_valt[i].squeeze(0) 

        file_name = os.path.join(output_dir, f"test_side_by_side_{i}.png")
        show_image(file_name,reference_image,generated_image,title_ref="Reference Image",title_gen="Generated Image")


    print(f"Saved side-by-side image for index {i}")
"""