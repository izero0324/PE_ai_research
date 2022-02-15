from torch.utils.data import Dataset
import os 
import torch 
import numpy as np 
import pydicom
import glob
import cv2

import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
#folder_name = list(glob.glob('/home/mel/PE_research/opendata/*'))
#random.shuffle(folder_name)
##print(folder_name)
#valid_size = 0.2
#num_patient = len(folder_name)
#split = int(valid_size*num_patient)
#train_folder = folder_name[split:]
#valid_folder = folder_name[:split]
#class0_path = []
#class1_path = []
#target0 = []
#target1 = []
#for x in train_folder:
#    #print(x)
#    class1 = glob.glob(x + '/1/*dcm')
#    class0 = glob.glob(x + '/0/*dcm')
#    for i in class1:
#        class1_path.append(i)
#        target1.append(1)
#    for i in class0:
#        class0_path.append(i)
#        target0.append(0)
#file_path = class0_path + class1_path
#target = target0 + target1
##print(file_path, target)
#train_data = zip(file_path, target)
#
#class0_path = []
#class1_path = []
#target0 = []
#target1 = []
#for x in valid_folder:
#    #print(x)
#    class1 = glob.glob(x + '/1/*dcm')
#    class0 = glob.glob(x + '/0/*dcm')
#    for i in class1:
#        class1_path.append(i)
#        target1.append(1)
#    for i in class0:
#        class0_path.append(i)
#        target0.append(0)
#file_path = class0_path + class1_path
#target = target0 + target1
##print(file_path, target)
#valid_data = zip(file_path, target)




#print(len(train_data), len(valid_data))
#max = 0
#min = 0
#for path, i in data:
#    a = pydicom.dcmread(path).pixel_array
#    if a.max() > max:
#        max = a.max()
#    if a.min() < min:
#        min = a.min()
#print(max, min)
def crop_center(img,cropx,cropy):
    x,y = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[startx:startx+cropx, starty:starty+cropy]

def dicom_loader(file):
    path, target = file
    #print(path+"!!!!!")
    #print(type(path))
    image = pydicom.dcmread(path).pixel_array
    #print(image)
    image[image>1624] = 1624
    #image = (image/2047.5)-1
    #image = (2*image/4095)-1
    image = (image/812)-1
    
    
    #print(image.max(), image.min())
    

    #print(img_tensor.shape)
    return image, target

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    rx = 0
    ry = 0
    if random_state is None:
        print('here')
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(random_state)
    rx = np.random.rand(400,400)
    ry = np.random.rand(400,400)

    #print(rx,ry)
    shape = image.shape
    dx = gaussian_filter((rx * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((ry * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    #print('xy',x,y)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    #print(indices)
    return map_coordinates(image, indices, order=1).reshape(shape)

class PE(Dataset):
    def __init__(self, file, transform=None):
        self.data = file
        self.transform = transform
        

    def __getitem__(self, index):

        data = self.data[index]

        img, target = dicom_loader(data)
        
        img = crop_center(img, 400, 400)

        img_tensor = torch.tensor(img, dtype = torch.float64)
        img_tensor = img_tensor.unsqueeze(0)
        #print(target[1])
        if self.transform is not None :
            img_tensor = self.transform(img_tensor.unsqueeze(0)).squeeze(0)
            #print(pos1, pos2)
            #if pos1 > 0.5:
            #    img = np.flip(img, axis = 0)
            #    img = torch.tensor(img.copy())
            #if pos2 > 0.5:
            #    img = np.flip(img, axis = 1)
            #    img = torch.tensor(img.copy())
        


        return img_tensor.float(), target.float()

    def __len__(self):
        return len(self.data)


class PE_transform(Dataset):
    def __init__(self, file, transform=None):
        self.data = file
        self.transform = transform

    def __getitem__(self, index):

        data = self.data[index]

        img, target = dicom_loader(data)
        
        img = crop_center(img, 180, 300)
        img = cv2.copyMakeBorder(img,110,110,50,50,cv2.BORDER_CONSTANT,value=0)
        #print(img[1])
        #if self.transform is not None and target[1] == 1:

        img_tensor = torch.tensor(img, dtype = torch.float64)
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.float(), target.float()

    def __len__(self):
        return len(self.data)

class PE_dinamic(Dataset):
    def __init__(self, file, transform=None):
        self.data = file
        self.transform = transform

    def __getitem__(self, index):

        data = self.data[index]
        path, target = data
        image = pydicom.dcmread(path).pixel_array.astype(np.int16)
        image = crop_center(image, 400, 400)
        img = image.copy()
        top_line,bottom_line, left_line,right_line = edgecut(image)
        img[img>1624] = 1624
        img = (img/812)-1
        img = img[top_line:400-bottom_line, left_line:400-right_line]
        img = cv2.copyMakeBorder(img,top_line,bottom_line, left_line,right_line,cv2.BORDER_CONSTANT,value=0)
        
        #print(img[1])
        #if self.transform is not None and target[1] == 1:

        img_tensor = torch.tensor(img, dtype = torch.float64)
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.float(), target.float()

    def __len__(self):
        return len(self.data)    
    
class mv_loader(Dataset):
    def __init__(self, file, transform=None):
        self.data = file
        self.transform = transform
        
    def __getitem__(self, index):
        data = self.data[index]
        path, target = data
        image = pydicom.dcmread(path).pixel_array
        image = crop_center(image, 400, 400)
        img = image.copy()
        top_line,bottom_line, left_line,right_line = edgecut(image)
        
        Lung_view = img.copy()
        M_view = img.copy()
        default_view = img.copy()
        
        default_view[default_view>1624] = 1624
        default_view = (default_view/812)-1
        default_view = default_view[top_line:400-bottom_line, left_line:400-right_line]
        default_view = cv2.copyMakeBorder(default_view,top_line,bottom_line, left_line,right_line,cv2.BORDER_CONSTANT,value=0)
        
        #L-400 W1500 lung
        Lung_view = Lung_view - 600
        Lung_view[Lung_view>1900] = 1900
        Lung_view[Lung_view<0] = 0
        Lung_view = (Lung_view/950)-1
        Lung_view = Lung_view[top_line:400-bottom_line, left_line:400-right_line]
        Lung_view = cv2.copyMakeBorder(Lung_view,top_line,bottom_line, left_line,right_line,cv2.BORDER_CONSTANT,value=0)
        
        #L40 W400 defult
        img = img - 950
        img[img>350] = 350
        img[img<0] = 0
        M_view = (M_view/175)-1
        M_view = M_view[top_line:400-bottom_line, left_line:400-right_line]
        M_view = cv2.copyMakeBorder(M_view,top_line,bottom_line, left_line,right_line,cv2.BORDER_CONSTANT,value=0)
        
        default_view_tensor = torch.tensor(default_view, dtype = torch.float64)
        default_view_tensor = default_view_tensor.unsqueeze(0)
        
        Lung_view_tensor = torch.tensor(Lung_view, dtype = torch.float64)
        Lung_view_tensor = Lung_view_tensor.unsqueeze(0)
        
        M_view_tensor = torch.tensor(M_view, dtype = torch.float64)
        M_view_tensor = M_view_tensor.unsqueeze(0)

        return default_view_tensor.float(), Lung_view_tensor.float(), M_view_tensor.float(), target.float()
    
    def __len__(self):
        return len(self.data)
    
class PE_cannyedge(Dataset):
    def __init__(self, file, transform=None):
        self.data = file
        self.transform = transform

    def __getitem__(self, index):

        data = self.data[index]
        path, target = data
        image = pydicom.dcmread(path).pixel_array
        
        img = image.copy()
        blurred = cv2.GaussianBlur(image, (1,1) , 0)
        blurred = ((blurred / np.max(blurred))*255).astype(np.uint8)
        canny = cv2.Canny(blurred, 30 , 70)
        img_mix = canny*0.8 + img*0.2
        img_canny = crop_center(img_mix, 180, 300)
        img_canny = cv2.copyMakeBorder(img_canny,110,110,50,50,cv2.BORDER_CONSTANT,value=0)
        
        #print(img[1])
        #if self.transform is not None and target[1] == 1:

        img_tensor = torch.tensor(img_canny, dtype = torch.float64)
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.float(), target.float()

    def __len__(self):
        return len(self.data)
def edgecut(image):
    
    #近似二值化dicom影像
    image [image >800] = 800
    image [image <700] = 700
    image  = image -700
    image  = (image /50)-1
    
    #尋找上下左右邊界
    i = 0
    j = 0
    left_edge = []
    top_edge = []
    right_edge = []
    bottom_edge = []
    
    for j in range(400):
        #print('y = ',j)
        flag = 0
        for i in range(400):

            if image[j,i]>0.9:
                flag = 1
                #print(j)

            if (image[j, i] < -0.9) and (flag == 1) :
                #print(i,j)
                left_edge.append(i)
                #y.append(1)
                break

    for j in range(400):
        #print('y = ',j)
        flag = 0
        for i in range(400):

            if image[j,399-i]>0.9:
                flag = 1
                #print(j)

            if (image[j, 399-i] < -0.9) and (flag == 1) :
                #print(i,j)
                right_edge.append(i)
                #y.append(1)
                break

    for j in range(400):
        #print('y = ',j)
        flag = 0
        for i in range(400):

            if image[i,j]>0.9:
                flag = 1
                #print(j)

            if (image[i, j] < -0.9) and (flag == 1) :
                #print(i,j)
                top_edge.append(i)
                #y.append(1)
                break

    for j in range(400):
        #print('y = ',j)
        flag = 0
        for i in range(400):

            if image[i,j]>0.9:
                flag = 1
                #print(j)

            if (image[i, j] < -0.9) and (flag == 1) :
                #print(i,j)
                top_edge.append(i)
                #y.append(1)
                break
    
    for j  in range(400):
        #print('y = ',j)
        flag = 0
        for i in range(400):

            if image[399-i,j]>0.9:
                flag =  1
                #print(j)

            if (image[399-i, j] < -0.9) and (flag == 1) :
                #print(i,j)
                bottom_edge.append(i)
                #y.append(1)
                break
    right_line = np.min(right_edge)         
    left_line = np.min(left_edge)
    top_line = np.min(top_edge)
    bottom_line = np.min(bottom_edge)
   
    

    return top_line, bottom_line, left_line, right_line