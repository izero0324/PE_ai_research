import os, sys, glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage.draw import polygon

def read_structure(structure):
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
        return contours

def get_mask(contours, slices):
    
    z = [round(s.ImagePositionPatient[2], 2) for s in slices]
    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]
    
    label = np.zeros_like(image, dtype=np.uint8)
    for con in contours:
        num = int(con['number'])
    for c in con['contours']:
        nodes = np.array(c).reshape((-1, 3))
        print(nodes)
        assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
        print(z)
        z_index = z.index(nodes[0, 2])
        r = (nodes[:, 1] - pos_r) / spacing_r
        c = (nodes[:, 0] - pos_c) / spacing_c
        rr, cc = polygon(r, c)
        label[rr, cc, z_index] = num

    colors = tuple(np.array([con['color'] for con in contours]) / 255.0)
    return label, colors


structure = pydicom.read_file('RS.16451260.dcm')
#print(structure.ROIContourSequence)
contours = read_structure(structure)
dcms = glob.glob(os.path.join('CTA_dicom', "*.dcm"))
slices = [pydicom.read_file(dcm) for dcm in dcms]
slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
image = np.stack([s.pixel_array for s in slices], axis=-1)

label, colors = get_mask(contours, slices)

label = label/2

import scipy.misc

for i in range(label.shape[2]):
    out = label[:,:,i]*255
    scipy.misc.imsave('outfile'+ str(i) +'.jpg', out)
# Plot to check slices, for example 50 to 59
#plt.figure(figsize=(15, 15))
#for i in range(9):
#    plt.subplot(3, 3, i + 1)
#    plt.imshow(image[..., i + 50], cmap="gray")
#    plt.contour(label[..., i + 50], levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors=colors)
#    plt.axis('off')