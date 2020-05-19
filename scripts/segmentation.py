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
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

class Seg:
    def __init__(self, img):
        # if isinstance(img,imread)==False:
        #     raise TypeError("The type of the 'img' entry parameter is wrong, use the 'readimage.imread' parameter.")
        self.img=img

    def meanshift(self,img,spatial,radiometric,minimum):
        pass

    def ms(self,n_samples=500,quantile=0.2,output_name=''):

        if output_name=='':
            st=os.path.basename(self.img.img_name).split('.')
            s=st[0]+'_mean_shift_'+str(n_samples)+"_"+str(quantile)+".tif"
            output_name=os.path.join(os.path.dirname(self.img.img_name),s)

        img=gdal.Open(self.img.img_name)
        arr=img.ReadAsArray()
        arr=cv2.merge(arr)
        shape=np.shape(arr)

        X = np.reshape(arr,(shape[0]*shape[1],shape[2]))

        bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        array=np.reshape(labels,(shape[0],shape[1]))
        print(f"number of estimated clusters : {len(np.unique(labels))}")
        self.img.new_img(array,output_name)

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



