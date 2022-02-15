import matplotlib.pyplot as plt
import pydicom
import numpy as np
from pydicom.data import get_testdata_files
from PIL import Image
import glob
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
#import SimpleITK as sitk


normal_list = list(glob.glob('/home/mel/PE_research/PE100/no_PE/*'))

for x in normal_list:
            name = x.split('/')[-1]
            image = list(glob.glob(x + '/0/*dcm'))
            for i in image:
                img_name = i.split('.')[0].split('/')[-1]
                #print(img_name)
                try:
                	data = pydicom.dcmread(i)
                	print(data)
                    #image = data.pixel_array

                except:
                    print('false')
    




