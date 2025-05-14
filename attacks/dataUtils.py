import os
from torchvision import transforms,utils,datasets
from torch.utils.data import Dataset,DataLoader
import random
import torch

cropH = 64
cropW = 64
# batchSize = 64
#加载数据集
def get_data_loaders(dataDir,batchSize):
    trainDir = os.path.join(dataDir,"train")
    testDir = os.path.join(dataDir,"test")
    
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
    
# 选择3200个数据来训练
#加载数据集
def get_sub_data_loaders(dataDir,batchSize,subDataset_size):
    trainDir = os.path.join(dataDir,"train")
    testDir = os.path.join(dataDir,"test")
    # sub_size = 3200
    
    data_transforms = {
        "train":transforms.Compose([
            transforms.RandomCrop((cropH,cropW),pad_if_needed=True),
            transforms.ToTensor(), #值域0-1，
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #值域 [-1，1]
        ]),
        
        "test":transforms.Compose([
            transforms.CenterCrop((cropH,cropW)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    }
    train_images = datasets.ImageFolder(trainDir,data_transforms["train"])
    indices_train = list(range(len(train_images)))
    indices_train = random.sample(indices_train,len(train_images))#按随机下标选取部分数据集
    sub_train_imgages = torch.utils.data.Subset(train_images, indices_train[:subDataset_size])    
    train_loader = DataLoader(sub_train_imgages,batch_size=batchSize,shuffle=True,drop_last =True,num_workers=4)
    
    test_images = datasets.ImageFolder(testDir,data_transforms["test"])
    indices_test = list(range(len(test_images)))
    indices_test = random.sample(indices_test,len(indices_test))#按随机下标选取部分数据集
    sub_test_imgages = torch.utils.data.Subset(train_images, indices_test[:subDataset_size])  
    #test_loader = DataLoader(sub_test_imgages,batch_size=batchSize,shuffle=True,drop_last =True,num_workers=4)
    test_loader = DataLoader(sub_test_imgages,batch_size=batchSize,shuffle=True,drop_last =True,num_workers=4)
    
    train_set_size = len(sub_train_imgages)
    test_set_size = len(sub_test_imgages)
    print("type of train_images ",type(train_images))
    print("type of train_loader ",type(train_loader))
    
    return train_loader,test_loader,train_set_size,test_set_size