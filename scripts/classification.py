import numpy as np
import cv2
import sys
import os
sys.path.insert(1, os.path.dirname(__file__))
from readimg import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans,MiniBatchKMeans
from shapely.geometry import mapping
import geopandas as gpd
from rasterio.mask import mask
import rasterio as rio
from sklearn.metrics import accuracy_score
import gdal


class classify:

    def __init__(self,img,bands=None):
        self.img=img
        if bands==None:
            bands=[i+1 for i in range(img.img.RasterCount)]
        self.bands=bands
        s=np.shape(rio.open(img.img_name).read())
        self.img.shape=[s[1],s[2],len(self.bands)]
        #print(self.img.shape)

    def kmeans(self):
        pass

    
    def randomForest(self,X,y,imOut,n_arvores=100,profundidade=2,X_test=None,y_test=None):

        clf = RandomForestClassifier(n_estimators=n_arvores, max_depth=profundidade,random_state=0)
        clf.fit(X, y) 

        srcIm=np.reshape(self.img.array,(self.img.shape[0]*self.img.shape[1],self.img.shape[2]))

        classificado=clf.predict(srcIm)
        outIm=np.reshape(classificado,(self.img.shape[0],self.img.shape[1]))
        
        self.img.new_img(outIm,imOut)

        if X_test is not None and y_test is not None:
            pred=clf.predict(X_test)
            return accuracy_score(y_test, pred)

        

    def get_samples(self,shp,nodata=-9999,trainTest=0,column='id'):
        img = rio.open(self.img.img_name)
        epsg=img.crs.to_epsg()
        gdf = gpd.read_file(shp)
        gdf=gdf.to_crs(epsg=epsg)

        X=[]
        y=[]

        for i in gdf.index:
            idx = gdf[column][i]
            geo = gdf['geometry'][i]
            extent_geojson = mapping(geo)
            im,_ = mask(img,
                        [extent_geojson],
                        nodata=nodata,
                        crop=True)
            no = [nodata]*np.shape(im)[0]
            no = np.array(no,dtype=im.dtype)
            im = cv2.merge(im)
            re = np.reshape(im,(np.shape(im)[0]*np.shape(im)[1],np.shape(im)[2]))
            mask_img = no!=re
            marr = np.ma.MaskedArray(re, mask=~mask_img)
            im = np.ma.compress_rows(marr)
            X += list(im)
            y += [idx]*len(im)

        if trainTest==0:
            return X,y
        elif trainTest>=1:
            raise Exception("Erro: A porcentagem deve ser maior do que 0 e menor do que 1.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=trainTest)
            return X_train, X_test, y_train, y_test