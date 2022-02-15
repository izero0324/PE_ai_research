import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import os
import numpy as np
import torch.nn as nn
from PIL import Image
from model.drn import drn_d_54,drn_d_base
from model.CBAM import drn_d_CBAM
from torchvision import datasets, transforms
import argparse
import pydicom
import torch.nn.functional as F 
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
random.seed(3)

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        out1 = x.to(device)
        for name, module in self.model._modules.items():
            #print(name)
            if name == 'model':
                out1 = module(out1)
            #if name in ['conv1', 'conv2', 'conv3']: 
            #    x = F.max_pool2d(x, 2,2)
            if name in self.target_layers:
                print(name)
                out1.register_hook(self.save_gradient)
                outputs += [out1]
        
        return outputs, out1

class ModelOutputs():

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, 'model')
        self.avgpool = nn.AdaptiveAvgPool2d(2)
    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        
        output = self.avgpool(output)
        print(output.shape)
        output = output.view(output.size(0), -1).to(device)
        output = self.model.fc(output)
        return target_activations, output

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
import os
def show_cam_on_image(img, mask, i, pred):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    #print(img.shape)
    img = np.transpose(img,(1,2,0))
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    name = i.split('.')[0].split('/')[-1]
    
    #file_name = os.path.join('0_1', name)
    
    file_name = os.path.join('sample/base/', name)
    print(file_name)
    cv2.imwrite(file_name + '_dicom.png', np.uint8(255 * img))
    
    cv2.imwrite(file_name +'_'+ name + "_cam.jpg", np.uint8(255 * cam))

class GradCam:
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()  
        self.extractor = ModelOutputs(self.model, target_layer_names)  
    def forward(self, input):
        return self.model(input)   
    def __call__(self, input, index = None):
        
        features, output = self.extractor(input)     
        if index == None:
            index = np.argmax(output.cpu().data.numpy())        
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        
        one_hot = torch.sum(one_hot.to(device) * output)     
        self.model.model.zero_grad()
        self.model.fc.zero_grad()
        one_hot.backward()
        #print(len(self.extractor.get_gradients()))
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()        
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        #print(target.shape)
        weights = grads_val[0, :]
        #weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)       
        for i, w in enumerate(weights):
        	cam += w * target[i, :, :]     
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (400, 400))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
	def __init__(self, model):
		self.model = model
		self.model.eval()
		

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model.model._modules.items():
			if module.__class__.__name__ == 'relu':
				self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		
		output = self.forward(input.to(device))

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		
		one_hot = torch.sum(one_hot.to(device) * output)

		# self.model.features.zero_grad()
		# self.model.classifier.zero_grad()
		one_hot.backward()

		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output



class CNNX(nn.Module):
    
    def __init__(self, backbone = 'drn', out_stride = 16, num_class = 2):
        super(CNNX, self).__init__()
        
        if backbone == 'drn':
            output_stride = 8
        #self.drn = drn_d_base(nn.BatchNorm2d)
        #self.drn = gc_drn_54(nn.BatchNorm2d)
        #self.model = mixnet_l()
        #self.model = drn_d_CBAM(nn.BatchNorm2d)
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


def load_model(Net, optimizer, model_file):
    assert os.path.exists(model_file),'There is no model file from'+model_file
    checkpoint = torch.load(model_file, map_location='cuda:0')
    Net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']+1
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return Net, optimizer, start_epoch

net = CNNX().to(device)
#print(net.model)
#net = CNNX().cpu()
resume = 'log/savemodel/0114_Fbest_acc_.pth'
#resume = 'best_acc_142.pth'
Net, _, _ = load_model(net, None, resume)
print(resume,'loaded')


import glob
#folder_name = list(glob.glob('/home/mel/PE_research/opendata/*'))
#
#folder_name.remove('/home/mel/PE_research/opendata/1')
##print(folder_name)
#folder1_name = list(glob.glob('/home/mel/PE_research/opendata/1/*'))
#
#random.shuffle(folder_name)
#random.shuffle(folder1_name)
#valid_size = 0.2
#test_size = 0.3
#num_patient = len(folder_name)
#num_patient1 = len(folder1_name)
##print(folder_name)
#num_patient = len(folder_name)
#num_patient1 = len(folder1_name)
#split = int(valid_size*num_patient)
#split1 = int(valid_size*num_patient1)
#
#split_2 = int(test_size*num_patient)
#split1_2 = int(test_size*num_patient1)
#train_folder = folder_name[split_2:] + folder1_name[split1_2:]
#valid_folder = folder_name[:split] + folder1_name[:split1]
#print(split1, split1_2)
##test_folder = folder_name[split:split_2] + folder1_name[split1:split1_2]
#test_folder = folder1_name[split1:split1_2]
#class0_path = []
#for x in test_folder:
#    #print(x)
#    class1 = glob.glob(x + '/1/*dcm')
#    class0 = glob.glob(x + '/0/*dcm')
#    for i in class0:
#        class0_path.append(i)
#patient = ['/home/mel/PE_research/PE100/PE_lung/NCKU/07685947', '/home/mel/PE_research/PE100/PE_lung/NCKU/17923011', '/home/mel/PE_research/PE100/PE_lung/NCKU/01006789', '/home/mel/PE_research/PE100/PE_lung/NCKU/07424280', '/home/mel/PE_research/PE100/PE_lung/NCKU/19284496', '/home/mel/PE_research/PE100/PE_lung/NCKU/19160682', '/home/mel/PE_research/PE100/PE_lung/NCKU/06552091', '/home/mel/PE_research/PE100/PE_lung/open/PAT004', '/home/mel/PE_research/PE100/PE_lung/open/PAT011', '/home/mel/PE_research/PE100/PE_lung/open/PAT017']
#patient = ['/home/mel/PE_research/PE100/non_PE/NCKU/01585175', '/home/mel/PE_research/PE100/non_PE/NCKU/18180554', '/home/mel/PE_research/PE100/non_PE/NCKU/05635111', '/home/mel/PE_research/PE100/non_PE/NCKU/19325176', '/home/mel/PE_research/PE100/non_PE/NCKU/11624858', '/home/mel/PE_research/PE100/non_PE/NCKU/17276749', '/home/mel/PE_research/PE100/non_PE/NCKU/14896060', '/home/mel/PE_research/PE100/non_PE/NCKU/03324841', '/home/mel/PE_research/PE100/non_PE/NCKU/01043033', '/home/mel/PE_research/PE100/non_PE/NCKU/07534808']
patient = ['/home/mel/PE_research/sample/']

path = []

for p in patient:

    path += list(glob.glob(p +'*dcm'))

grad_cam = GradCam(model = Net, \
				target_layer_names = ['layer8']) 
for i in path:
    print(i)
    img = pydicom.dcmread(i).pixel_array
    img[img>1624] = 1624
    input = np.float32(img/812)-1
    img = np.float32(img) / 1624
    
    input = preprocess_image(input)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None 
    mask = grad_cam(input, target_index)
    input = input.to(device)
    output = Net(input)
    pred = output.data.max(1, keepdim=True)[1]
    score = F.softmax(output)
    if pred == 1:
        print(score)
        img = crop_center(img, 400, 400)
        img = torch.tensor(img).float()
        img_rgb = img.unsqueeze_(0).repeat(3, 1, 1)    
        show_cam_on_image(img_rgb, mask, i, pred)
    
    #gb_model = GuidedBackpropReLUModel(model = Net)
    #gb = gb_model(input, index=target_index)
    ##utils.save_image(torch.from_numpy(gb), str(num) + '_gb.jpg')    
    #cam_mask = np.zeros(gb.shape)
    #for i in range(0, gb.shape[0]):
    #    cam_mask[i, :, :] = mask.transpose(1,0)    
    #cam_gb = np.multiply(cam_mask, gb)
        #utils.save_image(torch.from_numpy(cam_gb), str(num) + '_cam_gb.jpg')
        
    