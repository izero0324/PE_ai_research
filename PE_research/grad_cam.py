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
from torchvision.models import vgg16
from model.drn import drn_d_54
from torchvision import datasets, transforms
import argparse
import pydicom
import torch.nn.functional as F 
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self,extractor, model, target_layers):
        self.extractor = extractor
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        x = x.to(device)
        for name, module in self.model._modules.items():    
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model.attention, self.model.drn, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		#print(output.shape)
		output = self.model.avgpool(output)
		output = output.view(output.size(0), -1).to(device)
		#print(output.shape)
		output = self.model.fc(output)
		return target_activations, output

def preprocess_image(img):
    #print(img.shape)
    preprocessed_img = img.copy()
    #print(preprocessed_img.shape)
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0).unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]

def show_cam_on_image(img, mask, num):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    #print(img.shape)
    img = np.transpose(img,(1,2,0))
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(str(num) + 'dicom.jpg', np.uint8(255 * img))
    img1 = crop_center(img, 400, 400)
    cv2.imwrite(str(num) + 'dicom400.jpg', np.uint8(255 * img1))
    img2 = crop_center(img, 350, 350)
    cv2.imwrite(str(num) + 'dicom350.jpg', np.uint8(255 * img2))
    cv2.imwrite(str(num) + "_cam.jpg", np.uint8(255 * cam))

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
        self.model.drn.zero_grad()
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
        cam = cv2.resize(cam, (512, 512))
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
		for idx, module in self.model.drn._modules.items():
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
        self.conv4 = conv_set(32,128,3,1,0)
        self.conv5 = conv_set(128,classes,3,1,0)
    def forward(self, x):
        x = self.conv(x)
        out = self.conv1(x)
        out = F.max_pool2d(out, 2,2)
        out = self.conv2(out)
        out = F.max_pool2d(out, 2,2)
        out = self.conv3(out)
        out = F.max_pool2d(out, 2,2)
        out = self.conv4(out)
        out = self.conv5(out)
        out = F.max_pool2d(out, 2,2)
        return out 
        
class CNNX(nn.Module):
    
    def __init__(self, backbone = 'drn', out_stride = 16, num_class = 2):
        super(CNNX, self).__init__()
        
        if backbone == 'drn':
            output_stride = 8
        self.drn = drn_d_54(nn.BatchNorm2d)
        self.attention = A_net(512)
        self.avgpool = nn.AvgPool2d(36, stride = 1)
        self.fc = nn.Linear(430592, 2)
        


    def forward(self, input):
        x = self.drn(input)
        x = self.avgpool(x)
        
        atten = self.attention(input)
        atten = torch.sigmoid(atten)
        x = torch.mul(x, atten)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

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
#net = CNNX().cpu()
resume = 'log/savemodel/best_acc_.pth'
#resume = 'best_acc_142.pth'
Net, _, _ = load_model(net, None, resume)

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """    

    import glob

    file_name = list(glob.glob('/home/mel/PE_research/opendata/PAT010/0/*dcm'))
    file_name_sample = random.sample(file_name, k = 1)
    grad_cam = GradCam(model = Net, \
    				target_layer_names = ["layer8"]) 
    num = 1
    for i in file_name_sample:
        print(i)
        
        img = pydicom.dcmread(i).pixel_array
        img = np.float32(img) / 4095
        input = (img*2)-1
        input = preprocess_image(input)
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None 
        mask = grad_cam(input, target_index)
        img = torch.tensor(img).float()
        img_rgb = img.unsqueeze_(0).repeat(3, 1, 1)    
        show_cam_on_image(img_rgb, mask, num)
        
        gb_model = GuidedBackpropReLUModel(model = Net)
        gb = gb_model(input, index=target_index)
        #utils.save_image(torch.from_numpy(gb), str(num) + '_gb.jpg')    
        cam_mask = np.zeros(gb.shape)
        for i in range(0, gb.shape[0]):
            cam_mask[i, :, :] = mask.transpose(1,0)    
        cam_gb = np.multiply(cam_mask, gb)
        utils.save_image(torch.from_numpy(cam_gb), str(num) + '_cam_gb.jpg')
        num += 1
    