import sys, torch, random
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append("/home/rilea/exp0519/tuxiangchuli/task4")

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import transforms

from train.create_dataset import faceDataset
 

torch.set_default_tensor_type(torch.DoubleTensor)

writer = SummaryWriter("./logs")

import random, numpy
seed = random.randint(0, 65536)
torch.manual_seed(seed)     
random.seed(seed)
numpy.random.seed(seed)
print(seed)

train_img_dir = "/home/rilea/exp0519/tuxiangchuli/task4/data/train_img"
train_rating_path = "/home/rilea/exp0519/tuxiangchuli/task4/data/Rating_Collection/train_rating.csv"

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

batch_size_n = 4
epoch_n = 10

face_dataset = faceDataset(train_img_dir, train_rating_path, data_transform["train"],"train" )
trainloader = torch.utils.data.DataLoader(face_dataset, batch_size=batch_size_n, shuffle=True)



import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet50

net = resnet50(pretrained=True)


for param in net.parameters():
    param.requires_grad = False



inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 1) 


from torch.autograd import Variable
from torch.optim import lr_scheduler

criterion = nn.MSELoss()

# optimizer = optim.SGD(net.parameters(), lr=0.00004, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001 )
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for epoch in range(epoch_n):  # loop over the dataset multiple times

    print("epoch",epoch)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        img_name, img, rating_mean, rating_standard = data

        # 将这些数据转换成Variable类型
        # img, rating_standard = Variable(img) , Variable(rating_standard) 
        img, rating_mean = Variable(img) , Variable(rating_mean) 
         
  

        # rating_standard = rating_standard.view(4,-1)
        # 接下来就是跑模型的环节了，我们这里使用print来代替
        # print("epoch：", epoch, "的第", i, "个inputs", img.data.size(), "labels", rating_standard)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize.squeeze().size()
        outputs = net(img)
        outputs = torch.squeeze(outputs.view(1,-1))
        # print("outputs:", outputs)
        loss = criterion(outputs, rating_mean)
        loss.backward()
        optimizer.step()
        #
        # # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss ))
            running_loss = 0.0
        niter = epoch * len(data) + i
        writer.add_scalars('Train_loss', {'bs'+str(batch_size_n)+'_epoch'+str(epoch_n)+'_meandata_train_loss_Adam_pretrain_'+str(seed): loss.item()},
                           niter)

print('Finished Training')

PATH = './netargs/face_net__meandata_cpu_bs'+str(batch_size_n)+'_epoch'+str(epoch_n)+'_Adam_pretrain_'+str(seed)+'.pth'
torch.save(net.state_dict(), PATH)

print('save to',PATH)
