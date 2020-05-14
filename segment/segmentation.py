import numpy as np
import cv2
import gdal
import sys
import os
sys.path.insert(1, os.path.dirname(__file__))
from readimg import imread
from skimage import segmentation
from skimage.color import label2rgb
import os
import numpy as np

class Seg:
    def __init__(self, img):
        # if isinstance(img,imread)==False:
        #     raise TypeError("The type of the 'img' entry parameter is wrong, use the 'readimage.imread' parameter.")
        self.img=img

    def meanshift(self,img,spatial,radiometric,minimum):
        pass

    def SLIC(self,nSeg,compactness,sigma,outname='',export=True,labeled=True):
        if outname=='':
            os.path.join(os.path.dirname(self.img.img_name),'SLIC_nseg-'+str(nSeg)+"_compac-"+str(compactness)+"_sigma-"+str(sigma)+'.tif')
        ar=self.img.get_array([1,2,3])
        print(np.shape(ar))
        segments=segmentation.slic(ar,n_segments=nSeg,compactness=compactness,sigma=sigma) 
        if labeled:
            segments=label2rgb(segments, ar, kind='avg')
        if export:
            self.img.new_img(segments, outname)


# def SLIC(img):
#     img=gdal.Open(img)
#     segments=segmentation.slic(ar,n_segments=nSeg,compactness=compactness,sigma=sigma)
#     segments=label2rgb(segments, ar, kind='avg')



