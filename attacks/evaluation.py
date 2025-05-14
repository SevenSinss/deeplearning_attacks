" Author : taofangjian , Time : 2022-12-04 Description: Evaluation for target & untarget attacks."
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms,utils,datasets
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
from tqdm import *
import time
import random
import datetime
import codecs
import argparse

import dataUtils as data
from utils import *
from attacks import *
from models.AEmodels import *

import torchvision
import torchvision.transforms as T

import pytorch_ssim #[0,1]
from torch import optim
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 


#ImageSize
cropH = 64
cropW = 64
# dataDir = "E:/code/taofangjian/data/coco64/"
# dataDir = "E:/code/taofangjian/data/tiny-imagenet-200/"
dataDir = "/dev/shm/tiny-imagenet-200/"
# dataDir = "./data/tiny-imagenet-200/"
trainDir = os.path.join(dataDir,"train")
testDir = os.path.join(dataDir,"test")
batchSize = 64

expName = "Attack-Evaluation"
if not os.path.exists('./'+ expName):
    os.makedirs(expName)
img_dir = expName + "/EvaluationImg"
if not os.path.exists('./'+img_dir):
    os.makedirs(img_dir)
# UAPmodel_dir = expName + "/UAPmodel"
# if not os.path.exists('./'+UAPmodel_dir):
#     os.makedirs(UAPmodel_dir)
logfile = "Log-Attack-Evaluation-all.txt"
if not os.path.exists('./'+logfile):
    with codecs.open(logfile,'a+',encoding='utf-8') as f:
        f.write("")

parser = argparse.ArgumentParser(description='Input This Experiment Description.')
parser.add_argument('log', type=str, default="new \n",help='Input This Experiment Description.')

args = parser.parse_args()
print(args.log)
log = "Description:"+args.log +"\n"
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
with open('./'+logfile,"a") as file:
    file.write(expName + " " + now + ":\n"+log)
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Attack Target Model
path_smallmodel_tiny = "./modelgpu_tiny_all_60poch.pth"
AE_smallmodel_tiny = Make_model()
AE_smallmodel_tiny.load_state_dict(torch.load(path_smallmodel_tiny))
AE_smallmodel_tiny.to(device)
AE_smallmodel_tiny.eval()

# transfer Model
    # 0/100000 的样本重新训练
path_bigmodel_coco = "./2022-12-07 00-38_tmp58.pth"
AE_bigmodel_coco = Make_bigmodel()
AE_bigmodel_coco.load_state_dict(torch.load(path_bigmodel_coco))
AE_bigmodel_coco.to(device)
AE_bigmodel_coco.eval()

# ======== transfer Model retrain ========
decoderDir = "."
# decoderDir = "./Transfer-Evaluation-input_S"
    # 64/100000 的样本重新训练
path_bigdecoder_coco64 = decoderDir+"/transDecoder/subset_64/2022-12-07 22_32Bigmodel_decoder-epochs200.pth"
# path_bigdecoder_coco64 = decoderDir+"/transDecoder/subset_64/2022-12-08 21:42Bigmodel_decoder-epochs200.pth"
AE_bigdecoder_coco64 = Decoder_big()
AE_bigdecoder_coco64.load_state_dict(torch.load(path_bigdecoder_coco64))
AE_bigdecoder_coco64.to(device)
AE_bigdecoder_coco64.eval()
    # 128/100000 的样本重新训练
path_bigdecoder_coco128 = decoderDir+"/transDecoder/subset_128/2022-12-07 22_37Bigmodel_decoder-epochs200.pth"
# path_bigdecoder_coco128 = decoderDir+"/transDecoder/subset_128/2022-12-08 21:47Bigmodel_decoder-epochs200.pth"
AE_bigdecoder_coco128 = Decoder_big()
AE_bigdecoder_coco128.load_state_dict(torch.load(path_bigdecoder_coco128))
AE_bigdecoder_coco128.to(device)
AE_bigdecoder_coco128.eval()
    # 256/100000 的样本重新训练
path_bigdecoder_coco256 = decoderDir+"/transDecoder/subset_256/2022-12-07 22_43Bigmodel_decoder-epochs200.pth"
# path_bigdecoder_coco256 = decoderDir+"/transDecoder/subset_256/2022-12-08 21:52Bigmodel_decoder-epochs200.pth"
AE_bigdecoder_coco256 = Decoder_big()
AE_bigdecoder_coco256.load_state_dict(torch.load(path_bigdecoder_coco256))
AE_bigdecoder_coco256.to(device)
AE_bigdecoder_coco256.eval()
    # 512/100000 的样本重新训练
path_bigdecoder_coco512 = decoderDir+"/transDecoder/subset_512/2022-12-07 22_52Bigmodel_decoder-epochs200.pth"
# path_bigdecoder_coco512 = decoderDir+"/transDecoder/subset_512/2022-12-08 22:01Bigmodel_decoder-epochs200.pth"
AE_bigdecoder_coco512 = Decoder_big()
AE_bigdecoder_coco512.load_state_dict(torch.load(path_bigdecoder_coco512))
AE_bigdecoder_coco512.to(device)
AE_bigdecoder_coco512.eval()
    # 1024/100000 的样本重新训练
path_bigdecoder_coco1024 = decoderDir+"/transDecoder/subset_1024/2022-12-07 23_05Bigmodel_decoder-epochs200.pth"
# path_bigdecoder_coco1024 = decoderDir+"/transDecoder/subset_1024/2022-12-08 22:13Bigmodel_decoder-epochs200.pth"
AE_bigdecoder_coco1024 = Decoder_big()
AE_bigdecoder_coco1024.load_state_dict(torch.load(path_bigdecoder_coco1024))
AE_bigdecoder_coco1024.to(device)
AE_bigdecoder_coco1024.eval()

    # 2048/100000 的样本重新训练
path_bigdecoder_coco2048 = decoderDir+"/transDecoder/subset_2048/2022-12-07 23_29Bigmodel_decoder-epochs200.pth"
# path_bigdecoder_coco2048 = decoderDir+"/transDecoder/subset_2048/2022-12-08 22:33Bigmodel_decoder-epochs200.pth"
AE_bigdecoder_coco2048 = Decoder_big()
AE_bigdecoder_coco2048.load_state_dict(torch.load(path_bigdecoder_coco2048))
AE_bigdecoder_coco2048.to(device)
AE_bigdecoder_coco2048.eval()
    # 5000/100000 的样本重新训练
path_bigdecoder_coco5000 = decoderDir+"/transDecoder/subset_5000/2022-12-08 00_21Bigmodel_decoder-epochs200.pth"
# path_bigdecoder_coco5000 = decoderDir+"/transDecoder/subset_5000/2022-12-08 23:17Bigmodel_decoder-epochs200.pth"
AE_bigdecoder_coco5000 = Decoder_big()
AE_bigdecoder_coco5000.load_state_dict(torch.load(path_bigdecoder_coco5000))
AE_bigdecoder_coco5000.to(device)
AE_bigdecoder_coco5000.eval()

# ======== transfer Model retrain dataset enhanced ========

loss_history = []

# 高斯噪声
# noise = torch.empty(c_img_encode.size()).to(device)
# nn.init.normal_(noise,0,0.05)

# 白盒攻击PGD优化iters=10,epsilon=0.06,alpha=0.01

def evaluateNoise(dataDir,batchSize,AEmodel,noiseType):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device : ",device)
    
    train_loader,test_loader,train_set_size,test_set_size = data.get_sub_data_loaders(dataDir,batchSize,1024) # 选择1024个图像

    mse = torch.nn.MSELoss().to(device)   
    ssim = pytorch_ssim.SSIM().to(device)

    HccMSE,HccPSNR,HccSSIM,HccAPE,HssMSE,HssPSNR,HssSSIM,HssAPE = [],[],[],[],[],[],[],[]
    t = tqdm(test_loader)
    for images, _ in t: 
        images = images.to(device)
        input_C = images[0:images.shape[0] // 2] #input_C[32,3,64,64]
        input_S = images[images.shape[0] // 2:]
        C_encode,_ = AEmodel(input_S,input_C)
        
        noise = torch.empty(C_encode.size()).to(device)
        if noiseType == 'normal0.03':
            nn.init.normal_(noise,0,0.03)
            C_encode = C_encode + noise 
        elif noiseType == 'normal0.05':
            nn.init.normal_(noise,0,0.05)
            C_encode = C_encode + noise 
        elif noiseType == 'GB':
            transform = T.GaussianBlur(kernel_size=(3, 5), sigma=(0.5, 0.5))
            C_encode = transform(C_encode)
        S_decode_ori = AEmodel.decoder(C_encode)

        ccMSE = mse(((C_encode+ 1)/2),((input_C+ 1)/2)) 
        ccPSNR = psnr(ccMSE) 
        ccSSIM = ssim(((C_encode + 1)/2) , ((input_C + 1)/2)) #(N,) # [-1, 1] => [0, 1]
        ccAPE = APE(C_encode,input_C,batchSize) 

        ssMSE = mse(((S_decode_ori+ 1)/2),((input_S+ 1)/2))
        ssPSNR = psnr(ssMSE) 
        ssSSIM = ssim(((S_decode_ori + 1)/2) , ((input_S + 1)/2)) #(N,) # [-1, 1] => [0, 1]
        ssAPE = APE(S_decode_ori,input_S,batchSize) 

        HccMSE.append(ccMSE.item())
        HccPSNR.append(ccPSNR.item())
        HccSSIM.append(ccSSIM.item())
        HccAPE.append(ccAPE.item())
        HssMSE.append(ssMSE.item())
        HssPSNR.append(ssPSNR.item())
        HssSSIM.append(ssSSIM.item())
        HssAPE.append(ssAPE.item())

    print("[+] HccMSE = ",np.mean(HccMSE)," HccPSNR = ",np.mean(HccPSNR)," HccSSIM = ",np.mean(HccSSIM)," HccAPE = ",np.mean(HccAPE))
    print("[+] HssMSE = ",np.mean(HssMSE)," HssPSNR = ",np.mean(HssPSNR)," HssSSIM = ",np.mean(HssSSIM)," HssAPE = ",np.mean(HssAPE))
    log = "在tiny-image-net 测试集上测试攻击前的平均性能。 noiseType = "+ noiseType +"\n"
    log = log + "HccMSE = " + str(np.mean(HccMSE)) + " HccPSNR = " + str(np.mean(HccPSNR)) + " HccSSIM = " + str(np.mean(HccSSIM)) + " HccAPE = "+str(np.mean(HccAPE)) + "\n"
    log = log + "HssMSE = " + str(np.mean(HssMSE)) + " HssPSNR = " + str(np.mean(HssPSNR)) + " HssSSIM = " + str(np.mean(HssSSIM)) + " HssAPE = "+str(np.mean(HssAPE)) + "\n"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    with open('./'+logfile,"a") as file:
        file.write(expName + " " + now + ":\n"+log)


def evaluate(dataDir,batchSize,AEmodel):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device : ",device)
    
    train_loader,test_loader,train_set_size,test_set_size = data.get_sub_data_loaders(dataDir,batchSize,1024) # 选择1024个图像

    mse = torch.nn.MSELoss().to(device)   
    ssim = pytorch_ssim.SSIM().to(device)

    HccMSE,HccPSNR,HccSSIM,HccAPE,HssMSE,HssPSNR,HssSSIM,HssAPE = [],[],[],[],[],[],[],[]
    t = tqdm(test_loader)
    for images, _ in t: 
        images = images.to(device)
        input_C = images[0:images.shape[0] // 2] #input_C[32,3,64,64]
        input_S = images[images.shape[0] // 2:]
        C_encode,S_decode_ori = AEmodel(input_S,input_C)

        ccMSE = mse(((C_encode+ 1)/2),((input_C+ 1)/2)) 
        ccPSNR = psnr(ccMSE) 
        ccSSIM = ssim(((C_encode + 1)/2) , ((input_C + 1)/2)) #(N,) # [-1, 1] => [0, 1]
        ccAPE = APE(C_encode,input_C,batchSize) 

        ssMSE = mse(((S_decode_ori+ 1)/2),((input_S+ 1)/2))
        ssPSNR = psnr(ssMSE) 
        ssSSIM = ssim(((S_decode_ori + 1)/2) , ((input_S + 1)/2)) #(N,) # [-1, 1] => [0, 1]
        ssAPE = APE(S_decode_ori,input_S,batchSize) 

        HccMSE.append(ccMSE.item())
        HccPSNR.append(ccPSNR.item())
        HccSSIM.append(ccSSIM.item())
        HccAPE.append(ccAPE.item())
        HssMSE.append(ssMSE.item())
        HssPSNR.append(ssPSNR.item())
        HssSSIM.append(ssSSIM.item())
        HssAPE.append(ssAPE.item())

    print("[+] HccMSE = ",np.mean(HccMSE)," HccPSNR = ",np.mean(HccPSNR)," HccSSIM = ",np.mean(HccSSIM)," HccAPE = ",np.mean(HccAPE))
    print("[+] HssMSE = ",np.mean(HssMSE)," HssPSNR = ",np.mean(HssPSNR)," HssSSIM = ",np.mean(HssSSIM)," HssAPE = ",np.mean(HssAPE))
    log = "在tiny-image-net 测试集上测试攻击前的平均性能。\n"
    log = log + "HccMSE = " + str(np.mean(HccMSE)) + " HccPSNR = " + str(np.mean(HccPSNR)) + " HccSSIM = " + str(np.mean(HccSSIM)) + " HccAPE = "+str(np.mean(HccAPE)) + "\n"
    log = log + "HssMSE = " + str(np.mean(HssMSE)) + " HssPSNR = " + str(np.mean(HssPSNR)) + " HssSSIM = " + str(np.mean(HssSSIM)) + " HssAPE = "+str(np.mean(HssAPE)) + "\n"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    with open('./'+logfile,"a") as file:
        file.write(expName + " " + now + ":\n"+log)


def evaluateAttack(dataDir,batchSize,attackconfig,AEmodel,transferDecoder,modelStep,subsetSize):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device : ",device)
    
    train_loader,test_loader,train_set_size,test_set_size = data.get_sub_data_loaders(dataDir,batchSize,subsetSize)
    

    # 定义损失函数
    # S_mseloss = torch.nn.MSELoss().to(device)
    # C_mseloss = torch.nn.MSELoss().to(device) #计算明图重建损失
    mse = torch.nn.MSELoss().to(device)
    # ssim_loss = pytorch_ssim.SSIM().to(device)
    ssim = pytorch_ssim.SSIM().to(device)
    

    HccMSE,HccPSNR,HccSSIM,HccAPE,HssMSE,HssPSNR,HssSSIM,HssAPE,HstMSE,HstPSNR,HstSSIM,HstAPE = [],[],[],[],[],[],[],[],[],[],[],[]
    
    t = tqdm(test_loader)
    
        
    for images, _ in t: 
        images = images.to(device)
        input_C = images[0:images.shape[0] // 2] #input_C[32,3,64,64]
        input_S = images[images.shape[0] // 2:]
        C_encode,S_decode_ori = AEmodel(input_S,input_C)
        attack = Attack(attackconfig)
        # 攻击
        if attackconfig.istarget == False: # 无目标攻击
            if transferDecoder == None: 
                C_encode_pgd = attack.pgd_batch(C_encode,None,AEmodel.decoder).to(device)
            elif transferDecoder != None:# 黑盒迁移攻击
                C_encode_pgd = attack.pgd_batch(C_encode,None,transferDecoder).to(device)

            S_decode_pgd = AEmodel.decoder(C_encode_pgd)

        elif attackconfig.istarget == True: # 有目标攻击
            # 一个batch选择同一个目标攻击图像
            dataiter = iter(train_loader)
            images_test,_ = dataiter.next()
            targetImg = images_test[0].unsqueeze(0).to(device)
            targetImg = targetImg.repeat(32, 1, 1, 1) #复制32份，后面训练要计算loss,[32,3,64,64]
            targetImg.to(device)
            if transferDecoder == None:
                C_encode_pgd = attack.pgd_batch(C_encode,targetImg,AEmodel.decoder).to(device)
            elif transferDecoder != None:
                C_encode_pgd = attack.pgd_batch(C_encode,targetImg,transferDecoder).to(device)

#             if transferDecoder == None:
#                 C_encode_pgd = attack.pgd_batch_loss(C_encode,targetImg,AEmodel.decoder,"mse1").to(device)
#             elif transferDecoder != None:
#                 C_encode_pgd = attack.pgd_batch_loss(C_encode,targetImg,transferDecoder,"mse1").to(device)

            S_decode_pgd = AEmodel.decoder(C_encode_pgd)


        # 无目标攻击 评估C、C' , S'、S
        ccMSE = mse(((C_encode+ 1)/2),((C_encode_pgd+ 1)/2)) 
        ccPSNR = psnr(ccMSE) 
        ccSSIM = ssim(((C_encode + 1)/2) , ((C_encode_pgd + 1)/2)) #(N,) # [-1, 1] => [0, 1]
        ccAPE = APE(C_encode,C_encode_pgd,batchSize/2) # 实际的batchsize减半

        ssMSE = mse(((S_decode_ori+ 1)/2),((S_decode_pgd+ 1)/2))
        ssPSNR = psnr(ssMSE) 
        ssSSIM = ssim(((S_decode_ori + 1)/2) , ((S_decode_pgd + 1)/2)) #(N,) # [-1, 1] => [0, 1]
        ssAPE = APE(S_decode_ori,S_decode_pgd,batchSize/2) 

        # 有目标攻击  评估C、C' ,  S'、S ，S'、T
        if attackconfig.istarget == True:
            stMSE = mse(((targetImg+ 1)/2),((S_decode_pgd+ 1)/2))
            stPSNR = psnr(stMSE)
            stSSIM = ssim(((targetImg + 1)/2) , ((S_decode_pgd + 1)/2)) #(N,) # [-1, 1] => [0, 1]
            stAPE = APE(targetImg,S_decode_pgd,batchSize/2) 
            HstMSE.append(stMSE.item())
            HstPSNR.append(stPSNR.item())
            HstSSIM.append(stSSIM.item())
            HstAPE.append(stAPE.item())

        HccMSE.append(ccMSE.item())
        HccPSNR.append(ccPSNR.item())
        HccSSIM.append(ccSSIM.item())
        HccAPE.append(ccAPE.item())
        HssMSE.append(ssMSE.item())
        HssPSNR.append(ssPSNR.item())
        HssSSIM.append(ssSSIM.item())
        HssAPE.append(ssAPE.item())

    if attackconfig.istarget == False:
        if transferDecoder == None:
            log = "白盒攻击 untarget attck：\n"
            print("白盒 untarget attck ！")
        elif transferDecoder != None:
            log = "黑盒迁移攻击 untarget attck 模型重新训练数据集大小： "+str(modelStep)+"\n"
            print("黑盒迁移攻击 untarget attck ！ 模型重新训练数据集大小： ",modelStep)
        
        print("[+] HccMSE = ",np.mean(HccMSE)," HccPSNR = ",np.mean(HccPSNR)," HccSSIM = ",np.mean(HccSSIM)," HccAPE = ",np.mean(HccAPE))
        print("[+] HssMSE = ",np.mean(HssMSE)," HssPSNR = ",np.mean(HssPSNR)," HssSSIM = ",np.mean(HssSSIM)," HssAPE = ",np.mean(HssAPE))

        log = log + "无目标攻击，在"+str(dataDir)+"测试集上测试攻击后的平均性能。 扰动量"+str(attackconfig.eps)+" 迭代步数："+str(attackconfig.iters)+" 步幅："+str(attackconfig.alpha)+"\n"
        log = log + "HccMSE = " + str(np.mean(HccMSE)) + " HccPSNR = " + str(np.mean(HccPSNR)) + " HccSSIM = " + str(np.mean(HccSSIM)) + " HccAPE = " + str(np.mean(HccAPE)) + "\n"
        log = log + "HssMSE = " + str(np.mean(HssMSE)) + " HssPSNR = " + str(np.mean(HssPSNR)) + " HssSSIM = " + str(np.mean(HssSSIM)) + " HssAPE = " + str(np.mean(HssAPE)) + "\n"
    elif attackconfig.istarget == True:
        if transferDecoder == None:
            log = "白盒攻击 target attck ：\n"
            print("白盒 target attck ！")
        elif transferDecoder != None:
            log = "黑盒迁移攻击 target attck 模型重新训练数据集大小： "+str(modelStep)+"\n"
            print("黑盒迁移攻击 target attck ！ 模型重新训练数据集大小： ",modelStep)

        print("[+] HccMSE = ",np.mean(HccMSE)," HccPSNR = ",np.mean(HccPSNR)," HccSSIM = ",np.mean(HccSSIM)," HccAPE = ",np.mean(HccAPE))
        print("[+] HssMSE = ",np.mean(HssMSE)," HssPSNR = ",np.mean(HssPSNR)," HssSSIM = ",np.mean(HssSSIM)," HssAPE = ",np.mean(HssAPE))
        print("[+] HstMSE = ",np.mean(HstMSE)," HstPSNR = ",np.mean(HstPSNR)," HstSSIM = ",np.mean(HstSSIM)," HstAPE = ",np.mean(HstAPE))

        log = log + "有目标攻击，在"+str(dataDir)+"测试集上测试攻击后的平均性能。 扰动量："+str(attackconfig.eps)+" 迭代步数："+str(attackconfig.iters)+" 步幅："+str(attackconfig.alpha)+"\n"
        log = log + "HccMSE = " + str(np.mean(HccMSE)) + " HccPSNR = " + str(np.mean(HccPSNR)) + " HccSSIM = " + str(np.mean(HccSSIM)) + " HccAPE = " + str(np.mean(HccAPE)) + "\n"
        log = log + "HssMSE = " + str(np.mean(HssMSE)) + " HssPSNR = " + str(np.mean(HssPSNR)) + " HssSSIM = " + str(np.mean(HssSSIM)) + " HssAPE = " + str(np.mean(HssAPE)) + "\n"
        log = log + "HstMSE = " + str(np.mean(HstMSE)) + " HstPSNR = " + str(np.mean(HstPSNR)) + " HstSSIM = " + str(np.mean(HstSSIM)) + " HstAPE = " + str(np.mean(HstAPE)) + "\n"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    with open('./'+logfile,"a") as file:
        file.write(expName + " " + now + ":\n"+log)



        # ss_adv_Loss = S_mseloss(S_decode_ori,S_decode) #攻击后的解码图与容器图像中正常解码的密图之间的均方误差。

        # cc_ssim_loss = ssim(C_encode,C_encode_uap)
        # ss_ssim_loss = ssim(targetImg,S_decode)

        # now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
        
        # # 这里应该把 随机一张C'、C'+UAP、target、S_attack、UAP、S' 打印保存一下。
        # fig = plt.figure(figsize=(12,3)) #图像尺寸(W,H)

        # fig.add_subplot(1,5,1).axis('off') # C_test_encode TRUE
        # C_test_encode_cpu = C_test_encode.cpu().data[0]/2 +0.5  #
        # C_test_encode_np = C_test_encode_cpu.numpy()
        # # print('c_img_encode.shape:',C_test_encode_np.shape)
        # plt.imshow(np.transpose(C_test_encode_np,(1,2,0)))

        # fig.add_subplot(1,5,2).axis('off') # C_test_encode + UAP FAKE
        # Attacked_C_tensor = C_test_encode + UAP[0].unsqueeze(0)  #UAP(64,3,64,64) C_test_encode(1,3,64,64)
        # Attacked_C_np = Attacked_C_tensor.cpu().data[0]/2 +0.5
        # # Attacked_C_np = C_test_encode_np + UAP_cpu_np # >>> 这么写似乎有点问题，图像偏亮（多了几乎0.5像素），UAP是先按tensor(-1,1)计算的
        # # 所以上述应该是先tensor相加，再转为np，再除以2，而不是两个分别除以2再相加
        # plt.imshow(np.transpose(Attacked_C_np,(1,2,0)))

        # print("[C,C+UAP] mes:",mse(C_test_encode,Attacked_C_tensor).data," ssim:",ssim(C_test_encode,Attacked_C_tensor).data)  ## C_test_encode(1,3,64,64)

        # fig.add_subplot(1,5,3).axis('off') # S_decode TRUE
        # S_decode = AE_smallmodel_tiny.decoder(C_test_encode)
        # S_decode_cpu = S_decode.cpu().data[0]/2 +0.5 
        # S_decode_np = S_decode_cpu.numpy()
        # plt.imshow(np.transpose(S_decode_np,(1,2,0)))

        # fig.add_subplot(1,5,4).axis('off') # S_attack FAKE
        # S_attack = AE_smallmodel_tiny.decoder(Attacked_C_tensor) #Attacked_C_tensor(1,3,64,64)
        # S_attack_cpu = S_attack.cpu().data[0]/2 +0.5 
        # S_attack_np = S_attack_cpu.numpy()
        # plt.imshow(np.transpose(S_attack_np,(1,2,0)))

        # fig.add_subplot(1,5,5).axis('off') # Target image
        # # S_attack = AE_smallmodel_tiny.decoder(Attacked_C_tensor)
        # targetImg_cpu = targetImg.cpu().data[0]/2 +0.5 #targetImg(32,3,64,64)
        # targetImg_np = targetImg_cpu.numpy()
        # plt.imshow(np.transpose(targetImg_np,(1,2,0)))

        # print("[targetImg,decodeImg] mes:",mse(targetImg[0].unsqueeze(0),S_attack).data," ssim:",ssim(targetImg[0].unsqueeze(0),S_attack).data)

        # plt.savefig("./" +expName + "/" +  now + "_epoch"+str(epoch) +'.png')

        # loss_history.append(loss_AE)
        # print("[epoch = ",epoch+1,"] AEloss: ",np.mean(loss_AE),"ccLoss = ",np.mean(cLoss),"ssLoss = ",np.mean(sLoss),"ss_Loss_ssim = ",np.mean(ss_Loss_ssim),"cc_Loss_ssim = ",np.mean(cc_Loss_ssim))
        # log = "[epoch = " + str(epoch+1) + "] AEloss: " + str(np.mean(loss_AE)) +  " ccLoss = " + str(np.mean(cLoss)) + " ssLoss = " + str(np.mean(sLoss)) + " ss_Loss_ssim = " + str(np.mean(ss_Loss_ssim)) + " cc_Loss_ssim = " + str(np.mean(cc_Loss_ssim))+"\n"
        # now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        # with open('./'+logfile,"a") as file:
        #     file.write(expName + " " + now + ":\n"+log)


    # fig = plt.figure(figsize=(32,32)) #图像尺寸
    # plt.plot(loss_history)
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.savefig("./" + expName + "/lossHistory-"+now+".png")
#     plt.savefig( expName + "/lossHistory-AE-smalladv-coco64-4-29.png")


if __name__ == '__main__':


#     evaluate(dataDir,batchSize,AE_smallmodel_tiny) # 评估攻击前的性能
    
    
#     # 白盒无目标攻击
#     attackconfig_U = Config(eps=0.03,alpha=2/255,iters=50,istarget=False,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfig_U,AE_smallmodel_tiny,None,0,128)
#     attackconfig_U2 = Config(eps=0.06,alpha=2/255,iters=50,istarget=False,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfig_U2,AE_smallmodel_tiny,None,0,128)
#     # 白盒有目标攻击
#     pgdGoalConfig_T = Config(eps=0.03,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,pgdGoalConfig_T,AE_smallmodel_tiny,None,0,1024)
#     pgdGoalConfig_T2 = Config(eps=0.06,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,pgdGoalConfig_T2,AE_smallmodel_tiny,None,0,1024)
    
    
    # 白盒无目标攻击
#     attackconfig_U = Config(eps=0.03,alpha=0.5/255,iters=100,istarget=False,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfig_U,AE_smallmodel_tiny,None,0,128)
#     attackconfig_U2 = Config(eps=0.06,alpha=0.5/255,iters=100,istarget=False,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfig_U2,AE_smallmodel_tiny,None,0,128)
    # 白盒有目标攻击
#     pgdGoalConfig_T = Config(eps=0.03,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,pgdGoalConfig_T,AE_smallmodel_tiny,None,0,512)
#     pgdGoalConfig_T2 = Config(eps=0.06,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,pgdGoalConfig_T2,AE_smallmodel_tiny,None,0,512)

    # 高斯模糊进行无目标攻击
#     evaluateNoise(dataDir,batchSize,AE_smallmodel_tiny,"normal0.03") # 评估攻击前的性能
#     evaluateNoise(dataDir,batchSize,AE_smallmodel_tiny,"normal0.05") # 评估攻击前的性能
    evaluateNoise(dataDir,batchSize,AE_smallmodel_tiny,"GB") # 评估攻击前的性能

    
    # # 黑盒无目标迁移攻击
    
#     attackconfigTransfer_U = Config(eps=0.03,alpha=2/255,iters=100,istarget=False,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_U,AE_smallmodel_tiny,AE_bigmodel_coco.decoder,0,128)
#     attackconfigTransfer_U2 = Config(eps=0.06,alpha=2/255,iters=100,istarget=False,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_U2,AE_smallmodel_tiny,AE_bigmodel_coco.decoder,0,128)

    # # 黑盒有目标迁移攻击,黑盒有目标的攻击难度很大，所以做了部分查询重新训练 1024个样本
    # # 0/100000 的样本重新训练
    # attackconfigTransfer_T_0 = Config(eps=0.03,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_0,AE_smallmodel_tiny,AE_bigmodel_coco.decoder,0,1024)
    # attackconfigTransfer_T2_0 = Config(eps=0.06,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_0,AE_smallmodel_tiny,AE_bigmodel_coco.decoder,0,1024)

    # # 64/100000 的样本重新训练
    # attackconfigTransfer_T_64 = Config(eps=0.03,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_64,AE_smallmodel_tiny,AE_bigdecoder_coco64,64,1024)
    # attackconfigTransfer_T2_64 = Config(eps=0.06,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_64,AE_smallmodel_tiny,AE_bigdecoder_coco64,64,1024)

    # # 128/100000 的样本重新训练
    # attackconfigTransfer_T_128 = Config(eps=0.03,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_128,AE_smallmodel_tiny,AE_bigdecoder_coco128,128,1024)
    # attackconfigTransfer_T2_128 = Config(eps=0.06,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_128,AE_smallmodel_tiny,AE_bigdecoder_coco128,128,1024)
    # # 256/100000 的样本重新训练
    # attackconfigTransfer_T_256 = Config(eps=0.03,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_256,AE_smallmodel_tiny,AE_bigdecoder_coco256,256,1024)
    # attackconfigTransfer_T2_256 = Config(eps=0.06,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_256,AE_smallmodel_tiny,AE_bigdecoder_coco256,256,1024)
    # # 512/100000 的样本重新训练
    # attackconfigTransfer_T_512 = Config(eps=0.03,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_512,AE_smallmodel_tiny,AE_bigdecoder_coco512,512,1024)
    # attackconfigTransfer_T2_512 = Config(eps=0.06,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_512,AE_smallmodel_tiny,AE_bigdecoder_coco512,512,1024)
    # # 1024/100000 的样本重新训练
    # attackconfigTransfer_T_1024 = Config(eps=0.03,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_1024,AE_smallmodel_tiny,AE_bigdecoder_coco1024,1024,1024)
    # attackconfigTransfer_T2_1024 = Config(eps=0.06,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_1024,AE_smallmodel_tiny,AE_bigdecoder_coco1024,1024,1024)
    # # 2048/100000 的样本重新训练
    # attackconfigTransfer_T_2048 = Config(eps=0.03,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_2048,AE_smallmodel_tiny,AE_bigdecoder_coco2048,2048,1024)
    # attackconfigTransfer_T2_2048 = Config(eps=0.06,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_2048,AE_smallmodel_tiny,AE_bigdecoder_coco2048,2048,1024)
    # # 5000/100000 的样本重新训练
    # attackconfigTransfer_T_5000 = Config(eps=0.03,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_5000,AE_smallmodel_tiny,AE_bigdecoder_coco5000,5000,1024)
    # attackconfigTransfer_T2_5000 = Config(eps=0.06,alpha=2/255,iters=100,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
    # evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_5000,AE_smallmodel_tiny,AE_bigdecoder_coco5000,5000,1024)
   
# =========  调整迭代步幅和轮数 128个样本 =========  
    # 黑盒有目标迁移攻击,黑盒有目标的攻击难度很大，所以做了部分查询重新训练
    # 0/100000 的样本重新训练
#     attackconfigTransfer_T_0 = Config(eps=0.03,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_0,AE_smallmodel_tiny,AE_bigmodel_coco.decoder,0,128)
#     attackconfigTransfer_T2_0 = Config(eps=0.06,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_0,AE_smallmodel_tiny,AE_bigmodel_coco.decoder,0,128)

#     # 64/100000 的样本重新训练
#     attackconfigTransfer_T_64 = Config(eps=0.03,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_64,AE_smallmodel_tiny,AE_bigdecoder_coco64,64,128)
#     attackconfigTransfer_T2_64 = Config(eps=0.06,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_64,AE_smallmodel_tiny,AE_bigdecoder_coco64,64,128)

#     # 128/100000 的样本重新训练
#     attackconfigTransfer_T_128 = Config(eps=0.03,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_128,AE_smallmodel_tiny,AE_bigdecoder_coco128,128,128)
#     attackconfigTransfer_T2_128 = Config(eps=0.06,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_128,AE_smallmodel_tiny,AE_bigdecoder_coco128,128,128)
#     # 256/100000 的样本重新训练
#     attackconfigTransfer_T_256 = Config(eps=0.03,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_256,AE_smallmodel_tiny,AE_bigdecoder_coco256,256,128)
#     attackconfigTransfer_T2_256 = Config(eps=0.06,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_256,AE_smallmodel_tiny,AE_bigdecoder_coco256,256,128)
#     # 512/100000 的样本重新训练
#     attackconfigTransfer_T_512 = Config(eps=0.03,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_512,AE_smallmodel_tiny,AE_bigdecoder_coco512,512,128)
#     attackconfigTransfer_T2_512 = Config(eps=0.06,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_512,AE_smallmodel_tiny,AE_bigdecoder_coco512,512,128)
#     # 1024/100000 的样本重新训练
#     attackconfigTransfer_T_1024 = Config(eps=0.03,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_1024,AE_smallmodel_tiny,AE_bigdecoder_coco1024,1024,128)
#     attackconfigTransfer_T2_1024 = Config(eps=0.06,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_1024,AE_smallmodel_tiny,AE_bigdecoder_coco1024,1024,128)
#     # 2048/100000 的样本重新训练
#     attackconfigTransfer_T_2048 = Config(eps=0.03,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_2048,AE_smallmodel_tiny,AE_bigdecoder_coco2048,2048,128)
#     attackconfigTransfer_T2_2048 = Config(eps=0.06,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_2048,AE_smallmodel_tiny,AE_bigdecoder_coco2048,2048,128)
#     # 5000/100000 的样本重新训练
#     attackconfigTransfer_T_5000 = Config(eps=0.03,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T_5000,AE_smallmodel_tiny,AE_bigdecoder_coco5000,5000,128)
#     attackconfigTransfer_T2_5000 = Config(eps=0.06,alpha=0.3/255,iters=1000,istarget=True,target_label=None,learning_rate=None,showStepImg=False)
#     evaluateAttack(dataDir,batchSize,attackconfigTransfer_T2_5000,AE_smallmodel_tiny,AE_bigdecoder_coco5000,5000,128)

