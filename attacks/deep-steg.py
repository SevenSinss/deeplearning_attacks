import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms,utils,datasets
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
from tqdm import *
import time
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 


cropH = 64
cropW = 64
dataDir = "E:/code/taofangjian/data/COCO/"
#dataDir = "../data/tiny-imagenet-200/"
trainDir = os.path.join(dataDir,"train")
testDir = os.path.join(dataDir,"test")
batchSize = 64


def get_sub_data_loaders():
    
    sub_size = 3200
    
    data_transforms = {
        "train":transforms.Compose([
            transforms.RandomCrop((cropH,cropW),pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]),
        
        "test":transforms.Compose([
            transforms.CenterCrop((cropH,cropW)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    }
    train_images = datasets.ImageFolder(trainDir,data_transforms["train"])
    indices_train = list(range(len(train_images)))
    indices_train = random.sample(indices_train,len(train_images))
    sub_train_imgages = torch.utils.data.Subset(train_images, indices_train[:sub_size])    
    train_loader = DataLoader(sub_train_imgages,batch_size=batchSize,shuffle=True,drop_last =True,num_workers=4)
    
    test_images = datasets.ImageFolder(testDir,data_transforms["test"])
    indices_test = list(range(len(test_images)))
    indices_test = random.sample(indices_test,len(indices_test))
    sub_test_imgages = torch.utils.data.Subset(train_images, indices_test[:sub_size])  
    test_loader = DataLoader(sub_test_imgages,batch_size=batchSize,shuffle=False,drop_last =True,num_workers=4)
    
    train_set_size = len(sub_train_imgages)
    test_set_size = len(sub_test_imgages)

    
    return train_loader,test_loader,train_set_size,test_set_size


def get_data_loaders():
    data_transforms = {
        "train":transforms.Compose([
            transforms.RandomCrop((cropH,cropW),pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]),
        
        "test":transforms.Compose([
            transforms.CenterCrop((cropH,cropW)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    }
    train_images = datasets.ImageFolder(trainDir,data_transforms["train"])
    train_loader = DataLoader(train_images,batch_size=batchSize,shuffle=True,drop_last =True,num_workers=4)
    
    test_images = datasets.ImageFolder(testDir,data_transforms["test"])
    test_loader = DataLoader(test_images,batch_size=batchSize,shuffle=True,drop_last =True,num_workers=4)
    
    train_set_size = len(train_images)
    test_set_size = len(test_images)
    
    return train_loader,test_loader,train_set_size,test_set_size
    
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.conv1 = nn.Conv2d(3,50,3,padding = 1)
        self.conv2 = nn.Conv2d(3,10,4,padding = 1)
        self.conv3 = nn.Conv2d(3,5,5,padding = 2)
        
        self.conv4 = nn.Conv2d(65,50,3,padding = 1)
        self.conv5 = nn.Conv2d(65,10,4,padding = 1)
        self.conv6 = nn.Conv2d(65,5,5,padding = 2)
        
        self.conv7 = nn.Conv2d(68,50,3,padding = 1)
        self.conv8 = nn.Conv2d(68,10,4,padding = 1)
        self.conv9 = nn.Conv2d(68,5,5,padding = 2)
        
        self.conv10 = nn.Conv2d(65,50,3,padding = 1)
        self.conv11 = nn.Conv2d(65,10,4,padding = 1)
        self.conv12 = nn.Conv2d(65,5,5,padding = 2)
        
        self.conv13 = nn.Conv2d(65,50,3,padding = 1)
        self.conv14 = nn.Conv2d(65,10,4,padding = 1)
        self.conv15 = nn.Conv2d(65,5,5,padding = 2)
        
        self.conv16 = nn.Conv2d(65,50,3,padding = 1)
        self.conv17 = nn.Conv2d(65,10,4,padding = 1)
        self.conv18 = nn.Conv2d(65,5,5,padding = 2)
        
        self.conv19 = nn.Conv2d(65,50,3,padding = 1)
        self.conv20 = nn.Conv2d(65,10,4,padding = 1)
        self.conv21 = nn.Conv2d(65,5,5,padding = 2)
        
        self.conv22 = nn.Conv2d(65,3,3,padding = 1)
        
    def forward(self,input_S,input_C):
        x1 = F.relu(self.conv1(input_S))
        x2 = F.relu(self.conv2(input_S))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv3(input_S))
        x4 = torch.cat((x1,x2,x3),1) 
     
        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv6(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x4 = torch.cat((input_C,x4),1) 
        
        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat((x1,x2,x3),1)  
        
        x1 = F.relu(self.conv16(x4))
        x2 = F.relu(self.conv17(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv18(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv19(x4))
        x2 = F.relu(self.conv20(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv21(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        output = torch.tanh(self.conv22(x4))
        
        return output
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        
        self.conv1 = nn.Conv2d(3,50,3,padding = 1)
        self.conv2 = nn.Conv2d(3,10,4,padding = 1)
        self.conv3 = nn.Conv2d(3,5,5,padding = 2)
        
        self.conv4 = nn.Conv2d(65,50,3,padding = 1)
        self.conv5 = nn.Conv2d(65,10,4,padding = 1)
        self.conv6 = nn.Conv2d(65,5,5,padding = 2)
        
        self.conv7 = nn.Conv2d(65,50,3,padding = 1)
        self.conv8 = nn.Conv2d(65,10,4,padding = 1)
        self.conv9 = nn.Conv2d(65,5,5,padding = 2)
        
        self.conv10 = nn.Conv2d(65,50,3,padding = 1)
        self.conv11 = nn.Conv2d(65,10,4,padding = 1)
        self.conv12 = nn.Conv2d(65,5,5,padding = 2)
        
        self.conv13 = nn.Conv2d(65,50,3,padding = 1)
        self.conv14 = nn.Conv2d(65,10,4,padding = 1)
        self.conv15 = nn.Conv2d(65,5,5,padding = 2)
        
        self.conv16 = nn.Conv2d(65,3,3,padding = 1)
    
    def forward(self,x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv3(x))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv6(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        output = torch.tanh(self.conv16(x4))
        
        return output

class Make_model(nn.Module):
    def __init__(self):
        super(Make_model,self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self,input_S,input_C):
        output_Cprime = self.encoder(input_S,input_C)
        output_Sprime = self.decoder(output_Cprime)
        
        return output_Cprime,output_Sprime


class Encoder_big(nn.Module):
    def __init__(self):
        super(Encoder_big,self).__init__()
        # prep Network
        self.conv1 = nn.Conv2d(3,75,3,padding = 1)
        self.conv2 = nn.Conv2d(3,25,4,padding = 1)
        self.conv3 = nn.Conv2d(3,20,5,padding = 2)
        
        self.conv4 = nn.Conv2d(120,75,3,padding = 1)
        self.conv5 = nn.Conv2d(120,25,4,padding = 1)
        self.conv6 = nn.Conv2d(120,20,5,padding = 2)
        
        #Hiding Network 
        self.conv7 = nn.Conv2d(123,75,3,padding = 1)
        self.conv8 = nn.Conv2d(123,25,4,padding = 1)
        self.conv9 = nn.Conv2d(123,20,5,padding = 2)
        
        self.conv10 = nn.Conv2d(120,75,3,padding = 1)
        self.conv11 = nn.Conv2d(120,25,4,padding = 1)
        self.conv12 = nn.Conv2d(120,20,5,padding = 2)
        
        self.conv13 = nn.Conv2d(120,75,3,padding = 1)
        self.conv14 = nn.Conv2d(120,25,4,padding = 1)
        self.conv15 = nn.Conv2d(120,20,5,padding = 2)
        
        self.conv16 = nn.Conv2d(120,75,3,padding = 1)
        self.conv17 = nn.Conv2d(120,25,4,padding = 1)
        self.conv18 = nn.Conv2d(120,20,5,padding = 2)
        
        self.conv19 = nn.Conv2d(120,75,3,padding = 1)
        self.conv20 = nn.Conv2d(120,25,4,padding = 1)
        self.conv21 = nn.Conv2d(120,20,5,padding = 2)
        
        self.conv22 = nn.Conv2d(120,3,3,padding = 1)
        
    def forward(self,input_S,input_C):
        x1 = F.relu(self.conv1(input_S))
        x2 = F.relu(self.conv2(input_S))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv3(input_S))
        x4 = torch.cat((x1,x2,x3),1) 
        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv6(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x4 = torch.cat((input_C,x4),1) 
        
            # 进入hiding network
        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat((x1,x2,x3),1)  
        
        x1 = F.relu(self.conv16(x4))
        x2 = F.relu(self.conv17(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv18(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv19(x4))
        x2 = F.relu(self.conv20(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv21(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        output = torch.tanh(self.conv22(x4))
        
        return output
    
class Decoder_big(nn.Module):
    def __init__(self):
        super(Decoder_big,self).__init__()
        
        #解码器，重建密图
        self.conv1 = nn.Conv2d(3,75,3,padding = 1)
        self.conv2 = nn.Conv2d(3,25,4,padding = 1)
        self.conv3 = nn.Conv2d(3,20,5,padding = 2)
        
        self.conv4 = nn.Conv2d(120,75,3,padding = 1)
        self.conv5 = nn.Conv2d(120,25,4,padding = 1)
        self.conv6 = nn.Conv2d(120,20,5,padding = 2)
        
        self.conv7 = nn.Conv2d(120,75,3,padding = 1)
        self.conv8 = nn.Conv2d(120,25,4,padding = 1)
        self.conv9 = nn.Conv2d(120,20,5,padding = 2)
        
        self.conv10 = nn.Conv2d(120,75,3,padding = 1)
        self.conv11 = nn.Conv2d(120,25,4,padding = 1)
        self.conv12 = nn.Conv2d(120,20,5,padding = 2)
        
        self.conv13 = nn.Conv2d(120,75,3,padding = 1)
        self.conv14 = nn.Conv2d(120,25,4,padding = 1)
        self.conv15 = nn.Conv2d(120,20,5,padding = 2)
        
        self.conv16 = nn.Conv2d(120,3,3,padding = 1)
    
    def forward(self,x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv3(x))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv6(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat((x1,x2,x3),1) 
        
        output = torch.tanh(self.conv16(x4))
        
        return output

class Make_bigmodel(nn.Module):
    def __init__(self):
        super(Make_bigmodel,self).__init__()
        
        self.encoder = Encoder_big()
        self.decoder = Decoder_big()
        
    def forward(self,input_S,input_C):
        output_Cprime = self.encoder(input_S,input_C)
        output_Sprime = self.decoder(output_Cprime)
        
        return output_Cprime,output_Sprime
    

AEmodel =Make_bigmodel()
loss_history = []

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("device : ",device)
    

    AEmodel.to(device)

    
    train_loader,test_loader,train_set_size,test_set_size = get_sub_data_loaders()
    
    
    S_mseloss = torch.nn.MSELoss().to(device)
    C_mseloss = torch.nn.MSELoss().to(device) 
    
    optimizer = torch.optim.Adam(AEmodel.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)  
    
    
    for epoch in range(60):
        
        loss_all,c_loss,s_loss = [],[],[]
        t = tqdm(train_loader) 

        for images, _ in t: 
            images = images.to(device) 
            AEmodel.train() 
            
            with torch.enable_grad():
                optimizer.zero_grad() 
                

                input_C = images[0:images.shape[0] // 2]
                input_S = images[images.shape[0] // 2:]
            
                output_C,output_S = AEmodel(input_S,input_C)
                 
                
                #计算损失
                beta = 1.0
                ssLoss = S_mseloss(input_S,output_S)
                ccLoss = C_mseloss(input_C,output_C)
                loss =  beta * ssLoss + ccLoss
            
                loss.backward()
                optimizer.step()
            
                losses = {
                    "loss_all":loss.item(),
                    "ssLoss":ssLoss.item(),
                    "ccLoss":ccLoss.item()
                }
                loss_all.append(losses["loss_all"])
                c_loss.append(losses["ccLoss"])
                s_loss.append(losses["ssLoss"])
     
        loss_history.append(loss_all)
        print("[epoch = ",epoch+1,"] loss: ",np.mean(loss_all),"s_loss = ",np.mean(c_loss),"c_loss = ",np.mean(s_loss))
        

        scheduler.step(epoch)
        # save model
        path = "./modelgpubig_coco_all-60poch"+epoch+".pth"
        torch.save(AEmodel.state_dict(), path)
    
    fig = plt.figure(figsize=(8,8)) 
    plt.plot(loss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig("lossHistory-coco_all-60poch_big.png")



if __name__ == '__main__':
    
    train()
    
    

