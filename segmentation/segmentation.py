import numpy as np
import cv2
import gdal
import sys
import os
sys.path.insert(1, os.getcwd())
from segment.readimage import imread
from skimage import segmentation
from skimage.color import label2rgb
import os

class MeanShift:
    def __init__(self, img):
        if isinstance(img,imread)==False:
            raise TypeError("The type of the 'img' entry parameter is wrong, use the 'readimage.imread' parameter.")
        self.img=img

    def meanshift(self,img,spatial,radiometric,minimum):
        pass

    def SLIC(self,nSeg,compactness,sigma,outname='',export=True,labeled=True):
        if outname=='':
            os.path.join(os.path.dirname(self.img.img_name),'SLIC_nseg-'+str(nSeg)+"_compac-"+str(compactness)+"_sigma-"+str(sigma)+'.tif')
        self.img.array()
        segments=segmentation.slic(self.img.array(),n_segments=nSeg,compactness=compactness,sigma=sigma) 
        if labeled:
            segments=label2rgb(segments, self.img.array(), kind='avg')
        if export:
            self.img.new_img(segments, outname)

