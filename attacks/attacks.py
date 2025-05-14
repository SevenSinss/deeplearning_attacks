# whitebox attack
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import pytorch_ssim
from torch import optim
from tqdm import tqdm

# 先尝试普通的攻击，再试试产生通用对抗扰动，消除隐写

class Config():
    def __init__(self,eps,alpha,iters,istarget,target_label,learning_rate,showStepImg):
        self.eps = eps # 扰动范围最大量
        self.alpha = alpha # 扰动步幅
        self.iters = iters # 最大迭代轮数
        self.istarget = istarget # bool:是否为有目标攻击
        self.target_label = target_label # 攻击的目标标签
        self.learning_rate = learning_rate # 学习率
        # self.showStepImg = showStepImg #是否显示中间图

class Attack(object):
    def __init__(self,config):
        self.use_cuda = True
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.config = config
    def pgd_mode(self,images_c,target_img,decoder,mode):
        device = self.device
        decoder = decoder.to(self.device)
        img_c = images_c.to(device)
        mode = mode
        flag = 0
        if self.config.istarget:
            target_img = target_img.to(device)
#       
        img_s = decoder(img_c) # init copy
        ori_img_c = img_c.data
    
        if self.config.istarget==False:
            randoms = 2*torch.rand(img_c.shape)-1
            randoms.to(device)
            adv_image_c = img_c + self.config.alpha*randoms.sign().to(device)# 梯度方向上增大损失

            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_()
            
        ssim_loss = pytorch_ssim.SSIM().to(device) 
        mse_loss = torch.nn.MSELoss().to(device) #
        
        
        for i in range(self.config.iters):
            
            img_c = Variable(img_c)
            img_c.requires_grad = True 
            img_ss = decoder(img_c)

            decoder.zero_grad() 
            if self.config.istarget==False:
                cost = loss(img_s,img_ss).to(device)
            elif self.config.istarget==True:
                if mode == 2: 
                    if flag%2 == 0:
                        cost = ssim_loss(img_ss,target_img).to(device)
                    elif flag%2 == 1:
                        cost = -mse_loss(img_ss,target_img).to(device) 
                flag = flag +1
                if mode == 1:
                    loss1 = ssim_loss(img_ss,target_img).to(device)
                    loss2 = - mse_loss(img_ss,target_img).to(device) 
                    cost = loss1 + loss2

            cost.backward(retain_graph=True)
            adv_image_c = img_c + self.config.alpha*img_c.grad.sign() 
            
            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_() 

        if mode ==1:
            print("pgd mse_loss: ",loss2," ssim_loss:",loss1) 
        if mode ==2:
            if cost < 0:
                print("pgd mse_loss: ",cost)
            elif cost >0:
                print("pgd ssim_loss: ",cost)
            
        return img_c

    def pgd(self,images_c,target_img,decoder):
        device = self.device
        decoder = decoder.to(self.device)
        img_c = images_c.to(device)
        if self.config.istarget:
            target_img = target_img.to(device)
        img_s = decoder(img_c) # init copy
        ori_img_c = img_c.data
    
        if self.config.istarget==False:
            randoms = 2*torch.rand(img_c.shape)-1
            randoms.to(device)
            adv_image_c = img_c + self.config.alpha*randoms.sign().to(device)

            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_()
            
        ssim_loss = pytorch_ssim.SSIM().to(device) 
        mse_loss = torch.nn.MSELoss().to(device) #
        
        
        for i in range(self.config.iters):
            
            img_c = Variable(img_c)
            img_c.requires_grad = True 
            img_ss = decoder(img_c)

            decoder.zero_grad() 
            if self.config.istarget==False:
                cost = mse_loss(img_s,img_ss).to(device)
            elif self.config.istarget==True:
                
                loss1 = ssim_loss(img_ss,target_img).to(device)
                loss2 = - mse_loss(img_ss,target_img).to(device) 
                cost = loss1 + loss2

            cost.backward(retain_graph=True)

            adv_image_c = img_c + self.config.alpha*img_c.grad.sign() 
            
            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_() 
        print("pgd mse_loss: ",loss2," ssim_loss:",loss1) 
        return img_c

    def pgd_batch(self,images_c,target_img,decoder):
        device = self.device
        decoder = decoder.to(self.device)
        img_c = images_c.to(device)
        if self.config.istarget:
            target_img = target_img.to(device)

        img_s = decoder(img_c) # init copy
        ori_img_c = img_c.data
    
        if self.config.istarget==False:
            randoms = 2*torch.rand(img_c.shape)-1
            randoms.to(device)
            adv_image_c = img_c + self.config.alpha*randoms.sign().to(device)

            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_()
            
        ssim_loss = pytorch_ssim.SSIM().to(device) 
        mse_loss = torch.nn.MSELoss().to(device) #
        
        
        for i in range(self.config.iters):
            
            img_c = Variable(img_c)
            img_c.requires_grad = True 
            img_ss = decoder(img_c)

            decoder.zero_grad()
            if self.config.istarget==False:
                cost = mse_loss(img_s,img_ss).to(device)
            elif self.config.istarget==True:
                cost = -mse_loss(img_ss,target_img).to(device)

            cost.backward(retain_graph=True)

            adv_image_c = img_c + self.config.alpha*img_c.grad.sign() 
            
            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_() 
        print("pgd mse_loss: ",cost)    
        return img_c

    def pgd_batch_loss(self,images_c,target_img,decoder,lossfunction):
        device = self.device
        decoder = decoder.to(self.device)
        img_c = images_c.to(device)
        if self.config.istarget:
            target_img = target_img.to(device)

        img_s = decoder(img_c) # init copy
        ori_img_c = img_c.data

        if self.config.istarget==False:
            randoms = 2*torch.rand(img_c.shape)-1
            randoms.to(device)
            adv_image_c = img_c + self.config.alpha*randoms.sign().to(device)

            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_()
            
        ssim_loss = pytorch_ssim.SSIM().to(device) 
        mse_loss = torch.nn.MSELoss().to(device) #
        
        
        for i in range(self.config.iters):
            
            img_c = Variable(img_c)
            img_c.requires_grad = True
            img_ss = decoder(img_c)


            decoder.zero_grad() 
            if self.config.istarget==False:
                cost = mse_loss(img_s,img_ss).to(device)
            elif self.config.istarget==True:
                if lossfunction == "mes1":
                    cost = -mse_loss(img_ss,target_img).to(device)
                elif lossfunction == "mse2":
                    cost = -mse_loss(img_ss,target_img).to(device) + mse_loss(img_ss,img_s).to(device) 

            cost.backward(retain_graph=True)

            adv_image_c = img_c + self.config.alpha*img_c.grad.sign()
            
            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_() 
        print("pgd mse_loss: ",cost)    
        return img_c

    def pgd_bak(self,images_c,target_img,decoder):
        device = self.device
        decoder = decoder.to(self.device)
        img_c = images_c.to(device)
        if self.config.istarget:
            target_img = target_img.to(device)
        img_s = decoder(img_c) # init copy
        ori_img_c = img_c.data
    
        if self.config.istarget==False:
            randoms = 2*torch.rand(img_c.shape)-1
            randoms.to(device)
            adv_image_c = img_c + self.config.alpha*randoms.sign().to(device)

            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_()
            
        ssim_loss = pytorch_ssim.SSIM().to(device) 
        mse_loss = torch.nn.MSELoss().to(device) #
        
        print("[*] pgd_bak running ...")
        iters = self.config.iters
        for i in tqdm(range(iters)):
            
            img_c = Variable(img_c)
            img_c.requires_grad = True 
            img_ss = decoder(img_c)

            decoder.zero_grad() 
            if self.config.istarget==False:
                cost = mse_loss(img_s,img_ss).to(device)
            elif self.config.istarget==True:
                cost = -mse_loss(img_ss,target_img).to(device)

            cost.backward(retain_graph=True)

            adv_image_c = img_c + self.config.alpha*img_c.grad.sign() 
            
            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_() 
            
        print("pgd mse_loss: ",cost)    
        return img_c
        
    def cw(self,images_c,target_img):
        device = self.device
        img_c = images_c.to(device)
        if self.config.istarget == True:
            target_img = target_img.to(device)

        img_s = self.decoder(img_c) # init copy
        
        ori_img_c = img_c.data
        
        w = torch.zeros_like(img_c,requires_grad=True).to(device) 

        optimizer = torch.optim.Adam([w],lr=self.config.learning_rate) 
        Closs = torch.nn.MSELoss().to(device) #
        Closs2 = torch.nn.MSELoss().to(device)
        Sloss = torch.nn.MSELoss().to(device) #
        beta = 1.75 # 比重控制
        beta2 = 0.5
        for i in range(self.config.iters):
            c_cw = nn.Tanh()(w) 
            img_s_cw = self.decoder(c_cw)
            
            if(self.config.istarget == False):
                closs = Closs(img_c,c_cw)
                sloss = Sloss(img_s,img_s_cw)
                loss = (beta * closs) - sloss
            elif(self.config.istarget == True):
                closs = Closs(img_c,c_cw) - beta2 * Closs2(target_img,c_cw)
                sloss = Sloss(target_img,img_s_cw)
                loss = (beta * closs) + sloss
    
            optimizer.zero_grad() 
            loss.backward(retain_graph=True) 
            optimizer.step() 
            
            if self.config.istarget==False and sloss > 1.5 and closs < 0.01:
                print("[+] cw attack iterated ",i+1," times. Cost = ",loss," closs = ",closs," sloss = ",sloss)
                break
            elif self.config.istarget==True and sloss < 0.02 and closs < -1.5:
                print("[+] cw attack iterated ",i+1," times. Cost = ",loss," closs = ",closs," sloss = ",sloss)
                break
        print("cw loss:",loss," closs = ",closs," sloss = ",sloss)
        
        return c_cw
    
    def pgd_cost(self,images_c,target_img):
        device = self.device
        img_c = images_c.to(device)
        costs = []
        if self.config.istarget:
            target_img = target_img.to(device)

        img_s = self.decoder(img_c) # init copy
        ori_img_c = img_c.data
    
        if self.config.istarget==False:
            randoms = 2*torch.rand(img_c.shape)-1
            randoms.to(device)
            adv_image_c = img_c + self.config.alpha*randoms.sign().to(device)

            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_()
            
        
        loss = torch.nn.MSELoss().to(device) 
        
        for i in range(self.config.iters):
            
            img_c = Variable(img_c)
            img_c.requires_grad = True 
            img_ss = self.decoder(img_c)
            

            self.decoder.zero_grad() 
            if self.config.istarget==False:
                cost = loss(img_s,img_ss).to(device)
            elif self.config.istarget==True:
                cost = -loss(img_ss,target_img).to(device)
            
            cost.backward(retain_graph=True)
            costs.append(cost.item())
            adv_image_c = img_c + self.config.alpha*img_c.grad.sign() 
            
            eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
            img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_() 
        return img_c,costs    


class AttackFgsm(object):
    def __init__(self,config):
        self.use_cuda = True
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.config = config

    def fgsm_cost(self,images_c,images_s,decoder):
        device = self.device
        decoder = decoder.to(self.device)
        img_c = images_c.to(device)
        img_s = images_s.to(device)
        costs = []

        ori_img_c = img_c.data

        loss = torch.nn.MSELoss().to(device) 
        
        img_c = Variable(img_c)
        img_c.requires_grad = True 
        img_ss = decoder(img_c)

        decoder.zero_grad() 
        cost = loss(img_s,img_ss).to(device)
        cost.backward(retain_graph=True)
        
        costs.append(cost.item())
        if(self.config.eps == "random"):
            eps = random.uniform(0.01,0.06)
        else:
            eps = self.config.eps
        adv_image_c = img_c + eps*img_c.grad.sign()
        adv_image_c = torch.clamp(adv_image_c, min=-1, max=1).detach_()
        return adv_image_c,costs    

    def fgsm(self,images_c,decoder):
        device = self.device
        decoder = decoder.to(self.device)
        img_c = images_c.to(device)
        costs = []
        
        img_s = decoder(img_c) # init copy
        ori_img_c = img_c.data

        randoms = 2*torch.rand(img_c.shape)-1
        randoms.to(device)
        adv_image_c = img_c + (1/255)*randoms.sign().to(device)
        eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
        img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_()
        
        loss = torch.nn.MSELoss().to(device) 
        
        img_c = Variable(img_c)
        img_c.requires_grad = True 
        img_ss = decoder(img_c)

        decoder.zero_grad() 
        cost = loss(img_s,img_ss).to(device)
        cost.backward(retain_graph=True)
        
        costs.append(cost.item())
        adv_image_c = img_c + self.config.eps*img_c.grad.sign() 
        adv_image_c = torch.clamp(adv_image_c, min=-1, max=1).detach_() 
        return adv_image_c,costs    

    def fgsm_grad(self,images_c,decoder):
        device = self.device
        decoder = decoder.to(self.device)
        img_c = images_c.to(device)
        costs = []
        
        img_s = decoder(img_c) 
        ori_img_c = img_c.data

        
        randoms = 2*torch.rand(img_c.shape)-1
        randoms.to(device)
        adv_image_c = img_c + (1/255)*randoms.sign().to(device)
        eta = torch.clamp(adv_image_c - ori_img_c, min=-self.config.eps, max=self.config.eps) 
        img_c = torch.clamp(ori_img_c + eta, min=-1, max=1).detach_()
       
        
        loss = torch.nn.MSELoss().to(device) 
        
        img_c = Variable(img_c)# s_img_decode
        img_c.requires_grad = True 
        img_ss = decoder(img_c)

        decoder.zero_grad() 
        cost = loss(img_s,img_ss).to(device)
        cost.backward(retain_graph=True)
        
        costs.append(cost.item())
        adv_image_c = img_c + self.config.eps*img_c.grad.sign() 
        adv_image_c = torch.clamp(adv_image_c, min=-1, max=1).detach_() 
        return adv_image_c,img_c.grad    
