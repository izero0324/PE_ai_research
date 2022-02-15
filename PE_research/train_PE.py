from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F 
import time
import random
import datetime
import os
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import Tensor
import functools
from torchvision import datasets, transforms
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
import torchvision
from model.drn import drn_d_54, drn_d_base
from model.CBAM import drn_d_CBAM
from model.resnet import ResNet50 as resnet
from load_data import PE
from optimizer import Ranger
from collections import Counter
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score,accuracy_score, precision_score,f1_score


random.seed(1111)
#np.random.seed(3)
#torch.manual_seed(3)
#torch.cuda.manual_seed(3)
torch.backends.cudnn.deterministic=True



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)
test_size = 0.3
valid_size = 0.2
batch_size = 4
test_bs = 4

import glob

folder_name = list(glob.glob('/home/ubuntu/Andrew/PE_research/PE100/non_PE/NCKU/*'))

#print(folder_name)
folder1_name = list(glob.glob('/home/ubuntu/Andrew/PE_research/PE100/PE_lung/NCKU/*'))
folder1__name = list(glob.glob('/home/ubuntu/Andrew/PE_research/PE100/PE_lung/open/*'))
random.shuffle(folder_name)
random.shuffle(folder1_name)
random.shuffle(folder1__name)
#print(folder_name)
num_patient = 100
num_patient1 = len(folder1_name)
num__patient1 = len(folder1__name)
split = int(valid_size*num_patient)
split1 = int(valid_size*num_patient1)
split_2 = int(test_size*num_patient)
split1_2 = int(test_size*num_patient1)
split11 = int(valid_size*num__patient1)
split11_2 = int(test_size*num__patient1)

train_0 = folder_name[split_2:] + list(glob.glob('/home/ubuntu/Andrew/PE_research/PE100/non_PE/open/*'))
train_1 = folder1_name[split1_2:] + folder1__name[split11_2:]

#valid_0 = list(glob.glob('/home/ubuntu/Andrew/PE_research/1/non/*'))
#valid_1 = list(glob.glob('/home/ubuntu/Andrew/PE_research/1/PE/*'))

valid_0 = folder_name[:split]
valid_1 = folder1_name[:split1] + folder1__name[:split11]
#test_0 = folder_name[split:split_2]
#test_1 = folder1_name[split1:split1_2] + folder1__name[split11:split11_2]

test_0 = list(glob.glob('/home/ubuntu/Andrew/PE_research/1/non'))
test_1 = list(glob.glob('/home/ubuntu/Andrew/PE_research/1/PE/'))
print(len(train_0)+len(train_1), len(valid_1)+len(valid_0), len(test_1)+len(test_0))
print(test_1,test_0)


def get_data(folder0, folder1, repeat):
    class0_path = []
    class1_path = []
    target0 = []
    target1 = []
    for x in folder0:
        class0 = glob.glob(x + '/0/*dcm')
        for i in class0:
            class0_path.append(i)
            target0.append(torch.tensor([1.0,0.0]).long())
    for x in folder1:
        class1 = glob.glob(x + '/1/*dcm')
        for i in class1:
            for k in range(repeat):
                class1_path.append(i)
                target1.append(torch.tensor([0.0,1.0]).long())

        class0 = glob.glob(x + '/0/*dcm')
        for i in class0:
            class0_path.append(i)
            target0.append(torch.tensor([1.0,0.0]).long())
            #target1.append(torch.tensor(1))
        
            #target0.append(torch.tensor(0))
    #class0_path = random.sample(class0_path, k = len(class1_path))
    #target0 = random.sample(target0, k = len(target1))
    print(len(class0_path), len(class1_path))
    file_path = class0_path + class1_path
    target = target0 + target1
    #print(file_path, target)
    data = zip(file_path, target)
    return data, len(class0_path), len(class1_path)


train_data, _, _ = get_data(train_0, train_1, 3)
valid_data,pe0, pe1 = get_data(valid_0, valid_1, 1)
test_data, _, _ = get_data(test_0, test_1, 1)




Train_Set = PE(list(train_data), transform = None)
#Test_Set = CT_gray('Dataset/val', transform = None)
Valid_Set = PE(list(valid_data), transform = None)

Test_Set = PE(list(test_data), transform = None)


num_train = len(list(Train_Set))
num_valid = len(list(Valid_Set))
indices = list(range(num_train))
v_indices = list(range(num_valid))
#split = int(np.floor(valid_size*num_train))
#np.random.shuffle(indices)
#
train_idx, valid_idx = indices, v_indices
#train_idx=train_idx[:len(train_idx)//batch_size*batch_size]
#valid_idx=valid_idx[:len(valid_idx)//test_bs*test_bs]
#train_sampler = SubsetRandomSampler(train_idx)
#valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(Train_Set, batch_size=batch_size, shuffle = True,
                        num_workers=8, pin_memory=True)

valid_loader = DataLoader(Valid_Set, batch_size=test_bs, shuffle = False,
                      num_workers=8, pin_memory=True)
test_loader = DataLoader(Test_Set, batch_size=test_bs, shuffle = False,
                      num_workers=8, pin_memory=True)

class Focal_Loss(_Loss):
    def __init__(self, alpha = 3, gamma = 5, logits = False, reduce = False, ignore_index=10000, from_logits=False):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.logits = logits
        self.reduce = reduce
        self.loss  = nn.BCELoss(weight = torch.Tensor([3.0, 1.0]).to(device))
    def forward(self, inputs, targets):
        #CE = F.cross_entropy(inputs, targets, weight = torch.Tensor([1.0, 4.0]).to(device), ignore_index=self.ignore_index)
        inputs = torch.sigmoid(inputs)
        #print(inputs.shape, targets.shape)
        CE = self.loss(inputs, targets)
        pt = torch.exp(-CE)
        F_loss = self.alpha * (1-pt)**self.gamma * CE

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class Hswish(nn.Module):
    def forward(self, x):
        swish = F.relu6(x + 3 , inplace = True)
        return x* swish/6.

class conv_set(nn.Module):
    """docstring for conv_set"""
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(conv_set, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_ch),
            Hswish()
            )
    def forward(self, x):
        out = self.conv(x)
        return out

class A_net(nn.Module):

    def __init__(self, classes):
        super(A_net, self).__init__()
        self.classes = classes
        self.conv  = nn.Conv2d(1, 3, 1, 1, 0)
        self.conv1 = conv_set(3,12,3,1,0)
        self.conv2 = conv_set(12,16,3,1,0)
        self.conv3 = conv_set(16,32,3,1,0)
        self.conv4 = conv_set(32,classes,3,1,2)
        #self.conv5 = conv_set(64,classes,3,1,0)
    def forward(self, x):
        x = self.conv(x)
        out = self.conv1(x)
        out = F.max_pool2d(out, 2,2)
        out = self.conv2(out)
        out = F.max_pool2d(out, 2,2)
        out = self.conv3(out)
        out = F.max_pool2d(out, 2,2)
        out = self.conv4(out)
        return out 
        
class CNNX(nn.Module):
    
    def __init__(self, backbone = 'drn', out_stride = 16, num_class = 2):
        super(CNNX, self).__init__()
        
        if backbone == 'drn':
            output_stride = 8
        #self.drn = drn_d_base(nn.BatchNorm2d)
        #self.drn = gc_drn_54(nn.BatchNorm2d)
        #self.model = mixnet_l()
        self.model = resnet()
        #self.model = drn_d_CBAM(nn.BatchNorm2d)
        #self.attention = A_net(512)
        #self.conv_out = nn.Conv2d(512,2,1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.25)
        #self.fc = nn.Linear(2048, 2)
        self.fc = nn.Linear(2048, 2)
        


    def forward(self, input):
        x = self.model(input)
       #atten = self.attention(input)
       #atten = torch.sigmoid(atten)
       #x = torch.mul(x, atten)
        #x = self.conv_out(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self._dropout(x)

        return x



def save_model(state,save_model_path,modelname):
    filename = os.path.join(save_model_path,modelname+'_'+str(state['epoch']+1)+'.pth')
    torch.save(state,filename)

def save_best(state,save_model_path,modelname):
    filename = os.path.join(save_model_path,modelname+'_'+'.pth')
    torch.save(state,filename)

def load_model(Net, optimizer, model_file):
    assert os.path.exists(model_file),'There is no model file from'+model_file
    checkpoint = torch.load(model_file)
    print('load', model_file)
    Net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']+1
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return Net, optimizer, start_epoch

def train(model, data_loader, opt, loss, epoch,verbose = True):
    model.train()
    loss_avg = 0.0
    correct = 0
    for batch_idx, (data,target) in enumerate(data_loader):
        #print(data.shape)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss_avg = loss_avg + loss.item()
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        target = target[:,1].long()
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        verbose_step = len(data_loader) 
        if (batch_idx+1)  % verbose_step == 0 and verbose:
            print('Train Epoch: {}  Step [{}/{} ({:.0f}%)]  Loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch, batch_idx * len(data), len(train_idx),
                100. * batch_idx / len(train_idx), loss.item(), correct, len(train_idx),
            100. * correct / (len(train_idx))))
    return loss_avg / (len(train_idx))

def test(model, data_loader, loss):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        correct_pe = 0
        correct_normal = 0
        y = []
        y_score = []
        y_pred = []
        y_score = []
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            #print(target.shape)
            output = model(data)
            test_loss += loss_fn(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            target = target[:,1].long()
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            score = F.softmax(output, dim = 1)
            for i in range(target.shape[0]):
                y.append(target[i].cpu().numpy())
                y_score.append(score[i,1].cpu().numpy())
                y_pred.append(pred[i].cpu().numpy())
                if target[i] == 1:
                    correct_pe += pred[i].eq(target[i].data.view_as(pred[i])).cpu().sum().item()
    
                if target[i] == 0:
                    correct_normal += pred[i].eq(target[i].data.view_as(pred[i])).cpu().sum().item()

        test_loss /= len(data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), PE_acc: ({}), normal_acc: ({})\n'.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset), correct_pe, correct_normal))   
    return float(correct) / len(data_loader.dataset),y, y_score, y_pred


net = CNNX().to(device)

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.6, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        targets = targets[:,1]
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

class remix_loss(nn.Module):
    """docstring for remix_loss"""
    def __init__(self):
        super(remix_loss, self).__init__()
        self.loss1 = Focal_Loss()
        self.loss2 = TripletLoss()
    def forward(self, inputs, targets):
        loss = self.loss1(inputs, targets) + self.loss2(inputs, np.argmax(targets))

        return loss 
        

learning_rate = 0.001
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer = Ranger(net.parameters())

# Train the model
num_epochs = 50
model_name = 'drn'
loss_fn = Focal_Loss()
output_path = 'log'
resume = '/home/ubuntu/Andrew/PE_research/log/savemodel/0428_resbest_acc_.pth'

StartTime = time.time()
loss, val_acc, lr_curve = [], [], []

if resume is not None:
    net, optimizer, _ = load_model(net, optimizer, resume)
    #print('load', resume)

#'''
best_acc = 0.0
best = 0
for epoch in range(num_epochs):

    
    #if (epoch+1)%5 == 0:
    #    random.seed(epoch)
    #    train_data = get_train_data(train_folder)
    #    Train_Set = PE(list(train_data), transform = None)
    #    train_loader = DataLoader(Train_Set, batch_size=batch_size, shuffle = True,
    #                    num_workers=8, pin_memory=True)
    #    class0_path = random.sample(class0_path, k = 3*len(class1_path))
    #    target0 = random.sample(target0, k = 3*len(target1))
    #    print(len(class0_path), len(class1_path))
    #    file_path = class0_path + class1_path
    #    target = target0 + target1
    #    #print(file_path, target)
    #    train_data = zip(file_path, target)
    #    Train_Set = PE(list(train_data), transform = True)
    #    train_loader = DataLoader(Train_Set, batch_size=batch_size, shuffle = True,
    #                    num_workers=4, pin_memory=True)
    #lr = adjust_learning_rate(learning_rate, optimizer, epoch, epoch_list=[80, 170])
    train_loss = train(net, train_loader, optimizer, loss_fn, epoch, verbose=True)
    valid_acc, tp, tn = test(net, valid_loader, loss_fn)
    loss.append(train_loss)
    print((tn/pe0), tp/pe1)
    val_acc.append(valid_acc)
    if (epoch+1)%10 == 0 or epoch==num_epochs-1:
        save_model({'epoch':epoch,
                    'model_state_dict':net.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    },
                    os.path.join(output_path,'savemodel'),model_name)
    if tp/pe1 > 0.7 and tn>best:
    #if valid_acc > best_acc:
        print((tn//pe0), tp//pe1)
        print(valid_acc)
        save_best({'epoch':epoch,
                    'model_state_dict':net.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    },
                    os.path.join(output_path,'savemodel'),'0428_resbest_acc')
        best_acc = valid_acc
        best = tn
#'''

#net, _, _ = load_model(net, optimizer, '/home/mel/PE_research/log/savemodel/0428_resbest_acc.pth')
test_acc, y, y_score, y_pred = test(net, test_loader, loss_fn)
y = np.array(y)
y_pred = np.array(y_pred)
y_score = np.array(y_score)

precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
print('precision', precision, 'recall', recall)
ftr, tpr, _ = roc_curve(y, y_pred)
roc_auc = auc(ftr, tpr)
print('auc', roc_auc)

EndTime = time.time()
print('Time Usage: ', str(datetime.timedelta(seconds=int(round(EndTime-StartTime)))))



plt.figure()
plt.plot(loss)
plt.title('Train Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
plt.figure()
plt.plot(valid_acc)
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()

plt.figure()
plt.plot(val_acc)
plt.title('Valid Acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
