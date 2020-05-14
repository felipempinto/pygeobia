#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Felipe Matheus Pinto <felipematheuspinto27@gmail.com>

'''
This is an starting project
'''

import gdal
import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt

class imread:
    def __init__(self,img):
        self.img_name = img
        self.img = gdal.Open(img)
        self.img_array = gdal.ReadAsArray()

    def new_img(self, array, outname, driver='GTiff',dtype=None):
        '''
        This function can be used to create a new image using the same size of the input image,
        but who has passed by some processing and need to be exported. 

        This can acelerate some specific kinds of processing later in the project.

        Input parameters:
        - array: A numpy array, with 2D or 3D, 
        - outname: The output file name. It must have the file extension in the end of the name (e.g. '.tif').
        - driver: The type of the extension desired, to see all possible extension, check: https://gdal.org/drivers/raster/index.html
        - dtype: the gdal types of image. The default is the gdal.GDT_Float32. to se other options, check: https://naturalatlas.github.io/node-gdal/classes/Constants%20(GDT).html

        '''
        if dtype == 'float32':
            dtype = gdal.GDT_Float32        
        driver = gdal.GetDriverByName(driver)
        shape = np.shape(array)
        if len(shape) == 2:
            n = 1
        elif len(shape) == 3:
            n = shape[2]
        else:
            raise ValueError(f'The input array should be an 2D or 3D array, not an {len(shape)}D array')

        bands = cv2.split(array)
        proj = self.img.GetProjection()
        georef = self.img.GetGeoTransform()

        dst_file = driver.Create(outname,shape[1],shape[0],n,dtype)
        for i in range(self.img.GetRasterCount()):
            band = dst_file.GetRasterBand(i+1)
            band.WriteArray(bands[i])
        dst_file.SetProjection(proj)
        dst_file.SetGeoTransform(georef)
        dst_file.FlushCache()
        

    def plot_img(self):
        pass

    
        
