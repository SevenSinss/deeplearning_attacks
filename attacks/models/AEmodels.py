import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import transforms,utils,datasets
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
from tqdm import *
import time
import random

# 编写模型
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        # prep Network
        self.conv1 = nn.Conv2d(3,50,3,padding = 1)
        self.conv2 = nn.Conv2d(3,10,4,padding = 1)
        self.conv3 = nn.Conv2d(3,5,5,padding = 2)
        
        self.conv4 = nn.Conv2d(65,50,3,padding = 1)
        self.conv5 = nn.Conv2d(65,10,4,padding = 1)
        self.conv6 = nn.Conv2d(65,5,5,padding = 2)
        
        #Hiding Network 
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
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        
        #解码器，重建密图
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
    



class AdvNet(nn.Module):
    def __init__(self):
        super(AdvNet,self).__init__()
        
    #扰动网络
        self.conv1 = nn.Conv2d(3,50,3,padding = 1)
        self.conv2 = nn.Conv2d(3,10,4,padding = 1)
        self.conv3 = nn.Conv2d(3,5,5,padding = 2)
        
        self.conv4 = nn.Conv2d(65,50,3,padding = 1)
        self.conv5 = nn.Conv2d(65,10,4,padding = 1)
        self.conv6 = nn.Conv2d(65,5,5,padding = 2)
        
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
        
        output = torch.tanh(self.conv16(x4))
        
        return output    

class AdvNet_S(nn.Module):
    def __init__(self):
        super(AdvNet_S,self).__init__()
        
    #扰动网络
        self.conv1 = nn.Conv2d(3,3,3,padding = 1)
        self.conv2 = nn.Conv2d(3,2,4,padding = 1)
        self.conv3 = nn.Conv2d(3,1,5,padding = 2)
        
        
        self.conv16 = nn.Conv2d(6,3,3,padding = 1)
    
    def forward(self,x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x2 = F.pad(x2,(0,1,0,1),'constant',value=0) 
        x3 = F.relu(self.conv3(x))
        x4 = torch.cat((x1,x2,x3),1) 
        
        
        output = torch.tanh(self.conv16(x4))
        
        return output    

class AdvNet_S_new(nn.Module):
    def __init__(self):
        super(AdvNet_S_new,self).__init__()
        
    #扰动网络
        self.conv1 = nn.Conv2d(3,16,3,padding = 1)
        self.leak_relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.BN = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,3,3,padding = 1)
        
    def forward(self,x):
        
        x1 = self.leak_relu(self.BN(self.conv1(x)))
        x2 = self.conv2(x1) # 这里激活以后，全部是大于0的
        
        #output = torch.clamp(-1,1,x2)
        output = torch.tanh(x2)
#         output = x2
        return output   
    
class Encoder_big(nn.Module):
    def __init__(self):
        super(Encoder_big,self).__init__()
        # prep Network:
        self.conv1 = nn.Conv2d(3,75,3,padding = 1)
        self.conv2 = nn.Conv2d(3,25,4,padding = 1)
        self.conv3 = nn.Conv2d(3,20,5,padding = 2)
        
        self.conv4 = nn.Conv2d(120,75,3,padding = 1)
        self.conv5 = nn.Conv2d(120,25,4,padding = 1)
        self.conv6 = nn.Conv2d(120,20,5,padding = 2)
        
        #Hiding Network ：
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

# class Make_defmodel(nn.Mudule):
#     def __init__(self):
#         super(Make_defmodel,self).__init__()
        
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#         self.decoder_s = Decoder()
        
#     def forward(self,input_S,input_C):
#         output_Cprime = self.encoder(input_S,input_C)
#         output_
#         output_Sprime = self.decoder(output_Cprime)
        
        
#         return output_Cprime,output_Sprime

class Make_model_advsteg(nn.Module):
    def __init__(self):
        super(Make_model_advsteg,self).__init__()

        self.decoder = Decoder_big()

    def forward(self,input_C_adv): # 对抗调整后的C
        output_Sprime = self.decoder(input_C_adv)
        
        return output_Sprime    