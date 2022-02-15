from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F 
import time
import random
import datetime
import os
import numpy as np
import pydicom
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import Tensor
import functools
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
import torchvision
from model.drn import drn_d_54, drn_d_base
from model.CBAM import drn_d_CBAM
from model.resxnet import ig_resnext101_32x48d
from load_data import PE
from optimizer import Ranger
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score,accuracy_score, precision_score,f1_score



random.seed(1111)
#np.random.seed(3)
#torch.manual_seed(3)
#torch.cuda.manual_seed(3)
torch.backends.cudnn.deterministic=True



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
test_size = 0.3
valid_size = 0.2
batch_size = 8
test_bs = 1

import glob

import glob

folder_name = list(glob.glob('/home/mel/PE_research/PE100/non_PE/NCKU/*'))

#print(folder_name)
folder1_name = list(glob.glob('/home/mel/PE_research/PE100/PE_lung/NCKU/*'))
folder1__name = list(glob.glob('/home/mel/PE_research/PE100/PE_lung/open/*'))
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

train_0 = folder_name[split_2:] + list(glob.glob('/home/mel/PE_research/PE100/non_PE/open/*'))
train_1 = folder1_name[split1_2:] + folder1__name[split11_2:]
valid_0 = folder_name[:split]
valid_1 = folder1_name[:split1] + folder1__name[:split11]
test_0 = folder_name[split:split_2]
test_1 = folder1_name[split1:split1_2] + folder1__name[split11:split11_2]


def get_data(folder0, folder1, repeat):
    class0_path = []
    class1_path = []
    target0 = []
    target1 = []
    for x in folder0:
        id = x.split('/')[-1]
        place = x.split('/')[-2]
        class0 = glob.glob('/home/mel/PE_research/PE100/no_PE/'+ place + '/' + id + '/0/*dcm')
        for i in class0:
            class0_path.append(i)
            target0.append(torch.tensor([1.0,0.0]).long())
    for x in folder1:
        id = x.split('/')[-1]
        place = x.split('/')[-2]
        class1 = glob.glob('/home/mel/PE_research/PE100/PE/'+ place + '/' + id + '/1/*dcm')
        class0 = glob.glob('/home/mel/PE_research/PE100/PE/'+ place + '/' + id + '/0/*dcm')
        for i in class1:
            for k in range(repeat):
                class1_path.append(i)
                target1.append(torch.tensor([0.0,1.0]).long())

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


#train_data, _, _ = get_data(train_0, train_1, 3)
#valid_data,pe0, pe1 = get_data(valid_0, valid_1, 1)
test_data, _, _ = get_data(test_0, test_1, 1)

Test_Set = PE(list(test_data), transform = None)


#num_train = len(list(Train_Set))
#num_valid = len(list(Valid_Set))
#indices = list(range(num_train))
#v_indices = list(range(num_valid))
##split = int(np.floor(valid_size*num_train))
##np.random.shuffle(indices)
##
#train_idx, valid_idx = indices, v_indices
#train_idx=train_idx[:len(train_idx)//batch_size*batch_size]
#valid_idx=valid_idx[:len(valid_idx)//test_bs*test_bs]
#train_sampler = SubsetRandomSampler(train_idx)
#valid_sampler = SubsetRandomSampler(valid_idx)

#train_loader = DataLoader(Train_Set, batch_size=batch_size, shuffle = True,
#                        num_workers=8, pin_memory=True)
#
#valid_loader = DataLoader(Valid_Set, batch_size=test_bs, shuffle = True,
#                      num_workers=8, pin_memory=True)
test_loader = DataLoader(Test_Set, batch_size=test_bs, shuffle = False,
                      num_workers=8, pin_memory=True)

PE_list = test_1
normal_list = test_0
print(normal_list)


class Focal_Loss(_Loss):
    def __init__(self, alpha = 3, gamma = 5, logits = False, reduce = False, ignore_index=10000, from_logits=False):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.logits = logits
        self.reduce = reduce
        self.loss  = nn.BCELoss(weight = torch.Tensor([1.0, 5.0]).to(device))
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


        
class CNN_mix(nn.Module):
    
    def __init__(self, backbone = 'drn', out_stride = 16, num_class = 2):
        super(CNN_mix, self).__init__()
        
        if backbone == 'drn':
            output_stride = 8
        #self.drn = drn_d_base(nn.BatchNorm2d)
        #self.drn = gc_drn_54(nn.BatchNorm2d)
        self.model = mixnet_l()
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(1536, 2)
        


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

class CNN_lung(nn.Module):
    
    def __init__(self, backbone = 'drn', out_stride = 16, num_class = 2):
        super(CNN_lung, self).__init__()
        
        if backbone == 'drn':
            output_stride = 8
        #self.drn = drn_d_base(nn.BatchNorm2d)
        #self.drn = gc_drn_54(nn.BatchNorm2d)
        #self.model = mixnet_l()
        self.model = drn_d_CBAM(nn.BatchNorm2d)
        #self.attention = A_net(512)
        #self.conv_out = nn.Conv2d(512,2,1)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
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
        #x = self._dropout(x)

        return x

class CNN_drn(nn.Module):
    
    def __init__(self, backbone = 'drn', out_stride = 16, num_class = 2):
        super(CNN_drn, self).__init__()
        
        if backbone == 'drn':
            output_stride = 8
        #self.drn = drn_d_base(nn.BatchNorm2d)
        #self.drn = gc_drn_54(nn.BatchNorm2d)
        self.model = drn_d_base(nn.BatchNorm2d)
        #self.attention = A_net(512)
        #self.conv_out = nn.Conv2d(512,2,1)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
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
        #x = self._dropout(x)

        return x

class CNN_drn2(nn.Module):
    
    def __init__(self, backbone = 'drn', out_stride = 16, num_class = 2):
        super(CNN_drn2, self).__init__()
        
        if backbone == 'drn':
            output_stride = 8
        #self.drn = drn_d_base(nn.BatchNorm2d)
        #self.drn = gc_drn_54(nn.BatchNorm2d)
        self.model = drn_d_CBAM(nn.BatchNorm2d)
        #self.attention = A_net(512)
        #self.conv_out = nn.Conv2d(512,2,1)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
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
        #x = self._dropout(x)

        return x

class CNN_resX(nn.Module):
    
    def __init__(self, backbone = 'drn', out_stride = 16, num_class = 2):
        super(CNN_resX, self).__init__()
        
        if backbone == 'drn':
            output_stride = 8
        #self.drn = drn_d_base(nn.BatchNorm2d)
        #self.drn = gc_drn_54(nn.BatchNorm2d)
        self.model = ig_resnext101_32x48d()
        

        


    def forward(self, input):
        x = self.model(input)
       #atten = self.attention(input)
       #atten = torch.sigmoid(atten)
       #x = torch.mul(x, atten)
        #x = self.conv_out(x)


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



def test(model1, model2, model3, data_loader, loss):
    with torch.no_grad():
        model1.eval()
        model2.eval()
        model3.eval()
        test_loss = 0
        correct = 0
        correct_pe = 0
        correct_normal = 0
        y = []
        y_score = []
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            #print(target.shape)
            lung = model1(data)
            pred = lung.data.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()[0]
            if pred == 1:
                #output = 0.5*model2(data) + 0.5*model3(data)
                output = model2(data)
            else:
                output = lung
                
            test_loss += loss_fn(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            predict = pred.cpu().numpy()[0]
            #if predict == 1:
            #    output = model3(data)
            #    pred = output.data.max(1, keepdim=True)[1]
            target = target[:,1].long()
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            
            for i in range(target.shape[0]):
                y.append(target[i].cpu().numpy())
                y_score.append(pred[i].cpu().numpy())
                if target[i] == 1:
                    correct_pe += pred[i].eq(target[i].data.view_as(pred[i])).cpu().sum().item()
    
                if target[i] == 0:
                    correct_normal += pred[i].eq(target[i].data.view_as(pred[i])).cpu().sum().item()

        test_loss /= len(data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), PE_acc: ({}), normal_acc: ({})\n'.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset), correct_pe, correct_normal))   
    return float(correct) / len(data_loader.dataset), y, y_score 

def preprocess_image(img):
    #print(img.shape)
    preprocessed_img = img.copy()
    #print(preprocessed_img.shape)
    preprocessed_img = crop_center(img, 400, 400)
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0).unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    #print(input.shape)
    return input

def crop_center(img,cropx,cropy):
    x,y = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[startx:startx+cropx, starty:starty+cropy]

def per_patient(model1, model2, model3, PE, non_PE):
    with torch.no_grad():
        model1.eval()
        model2.eval()
        model3.eval()
        PE_correct = 0
        normal_correct = 0
        for x in PE:
            name = x.split('/')[-1]
            place = x.split('/')[-2]
            image = []
            class1 = glob.glob('/home/mel/PE_research/PE100/PE/'+ place + '/' + name + '/1/*dcm')
            class0 = glob.glob('/home/mel/PE_research/PE100/PE/'+ place + '/' + name + '/0/*dcm')
            image = class1 + class0
            cnt = 0
            for i in image:
                img = pydicom.dcmread(i).pixel_array
                img[img>1624] = 1624
                img = np.float32(img/812)-1
                img = preprocess_image(img).to(device)
                lung = model1(img)
                pred = lung.data.max(1, keepdim=True)[1]
                pred = pred.cpu().numpy()[0]
                if pred == 1:
                    #output = 0.5*model2(img) + 0.5*model3(img)
                    output = model2(img)
                    pred = output.data.max(1, keepdim=True)[1]
                    pred = pred.cpu().numpy()[0]
                    if pred == 1:
                        cnt += 1
            print(name + 'PE contain:', cnt)
            if cnt>0 :
                PE_correct += 1
        print('normal')
        for x in non_PE:
            name = x.split('/')[-1]
            image = []
            place = x.split('/')[-2]
            class0 = glob.glob('/home/mel/PE_research/PE100/no_PE/'+ place + '/' + name + '/0/*dcm')
            image = class0
            cnt = 0
            false_list = []
            for i in image:
                img_name = i.split('.')[0].split('/')[-1]
                #print(img_name)
                try:
                    img = pydicom.dcmread(i).pixel_array
                except:
                    print(name, 'false')
                img[img>1624] = 1624
                img = np.float32(img/812)-1
                img = preprocess_image(img).to(device)
                lung = model1(img)
                pred = lung.data.max(1, keepdim=True)[1]
                pred = pred.cpu().numpy()[0]
                if pred == 1:
                    #output = 0.5*model2(img) + 0.5*model3(img)
                    output = model2(img)
                    pred = output.data.max(1, keepdim=True)[1]
                    pred = pred.cpu().numpy()[0]
                    if pred == 1:
                        cnt += 1
                        false_list.append(img_name)
            print(name + 'PE contain:', cnt)
            print(false_list)
            
            if cnt ==0:
                normal_correct +=1

    return PE_correct, normal_correct




net1 = CNN_lung().to(device)



net2 = CNN_drn2().to(device)

net3 = CNN_drn().to(device)
        

learning_rate = 0.001
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer1 = Ranger(net1.parameters())
optimizer2 = Ranger(net2.parameters())
optimizer3 = Ranger(net3.parameters())


StartTime = time.time()



loss_fn = Focal_Loss()

net1, _, _ = load_model(net1, optimizer1, 'log/savemodel/0114_Lbest_acc_.pth')
net2, _, _ = load_model(net2, optimizer2, 'log/savemodel/0115_Fbest_acc_.pth')
net3, _, _ = load_model(net3, optimizer3, 'log/savemodel/0114_Fbest_acc_.pth')

test_acc, y, y_score = test(net1, net2, net3, test_loader, loss_fn)
EndTime = time.time()
print('Time Usage: ', str(datetime.timedelta(seconds=int(round(EndTime-StartTime)))))

y = np.array(y)
y_score = np.array(y_score)
#print(y.shape, y_score.shape)
#print(y)
precision = precision_score(y, y_score, average='weighted')
recall = recall_score(y, y_score, average='weighted')
print('precision', precision, 'recall', recall)
ftr, tpr, _ = roc_curve(y, y_score)
roc_auc = auc(ftr, tpr)
print('auc', roc_auc)
plt.figure()
lw = 2
plt.plot(ftr, tpr, color='darkorange', lw=lw,
         label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


PE_correct, normal_correct = per_patient(net1, net2, net3, PE_list, normal_list)
print(PE_correct, normal_correct)





