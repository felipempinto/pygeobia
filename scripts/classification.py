import numpy as np
import cv2
import gdal
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


    def get_samples(self,vector_file,column='id',trainTest=0.0,noData=256):
        img=self.img.img_name
        shapefile = gpd.read_file(vector_file)
        X=[]
        y=[]
        #cont=1
        #total=len(shapefile)
        for i in range(len(shapefile['geometry'])):
            with rio.open(img) as imgOriginal:
                try:
                    extent_geojson = mapping(shapefile['geometry'][i])
                except AttributeError:
                    print(f"Geometry {i} with problems")
                else:
                    imgRec,_= mask(imgOriginal,
                                   [extent_geojson],
                                   nodata=noData,
                                   crop=True)
                    
                    XTemp=[]

                    shape=np.shape(imgRec)
                    merge=cv2.merge(imgRec)
                    re=np.reshape(merge,(shape[1]*shape[2],shape[0]))
                    no=np.array([noData]*shape[0],dtype=re.dtype)
                    re=re[re!=no]
                    XTemp=np.reshape(re,(int(np.shape(re)[0]/shape[0]),shape[0]))

            for l in XTemp:
                # if l not in X:
                X.append(l)
                y.append(shapefile[column][i])
            #print(f"Loop {cont} of {total} done!")
            #cont+=1
        if trainTest==0:
            return X,y
        elif trainTest>=1:
            raise Exception("Error: you need to pass a value between 0 and 1.")
        else:
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=trainTest)
            # return X_train, X_test, y_train, y_test
            return train_test_split(X, y, test_size=trainTest)

        