import sys, torch, random
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append("/home/rilea/exp0519/tuxiangchuli/task4")

 
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib as plt

from torchvision.models import resnet50
from train.create_dataset import faceDataset
import torch.nn as nn


with torch.no_grad():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }



    test_img_dir = "/home/rilea/exp0519/tuxiangchuli/task4/data/test_img"
    test_rating_path = "/home/rilea/exp0519/tuxiangchuli/task4/data/Rating_Collection/test_rating.csv"

    test_dataset = faceDataset(test_img_dir, test_rating_path, data_transform["train"],"test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    dataiter = iter(testloader)
    img_name, images, labels, rating_nom = dataiter.next()

    print('GroundTruth: ', ' '.join('%5s' % img_name[j] for j in range(8)))
    print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(8)))


    # PATH = './face_net.pth'train/face_net_bs_4_epoch_2.pth
    # PATH = '/home/rilea/exp0519/tuxiangchuli/task4/train/netargs/face_net_cpu_bs_4_epoch_1.pth'
    PATH = './netargs/face_net__meandata_cpu_bs4_epoch10_Adam_pretrain_50605.pth'
    net = resnet50()
    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, 1)
    net.load_state_dict(torch.load(PATH))
    net.eval() 

    outputs = net(images)
    print(outputs)
