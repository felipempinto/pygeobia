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


class classify:

    def __init__(self,img,bands=None):
        self.img=img
        if bands==None:
            bands=[i+1 for i in range(img.img.RasterCount)]
        self.bands=bands

    def kmeans(self):
        pass

    def get_samples(self)

    
    def randomForest(self,X,y,imOut,n_arvores=100,profundidade=2):

        clf = RandomForestClassifier(n_estimators=n_arvores, max_depth=profundidade,random_state=0)
        clf.fit(X, y) 

        srcIm=np.reshape(self.img.array,(self.img.shape[0]*self.img.shape[1],self.img.nBandas))

        classificado=clf.predict(srcIm)
        outIm=np.reshape(classificado,(self.img.shape[0],self.img.shape[1]))
        
        self.img.newIm(outIm,imOut)

        if acuracia==True:
            if X_test==None:
                raise Exception("Insira valores para X_test")
            if y_test==None:
                raise Exception("Insira valores para y_test")
            pred=clf.predict(X_test)
            print("Acurácia da classificação: ",accuracy_score(y_test, pred))


    def coletaDeAmostras(self,vector_file,trainTest=0.0,noData=256):
        img=self.img.img_name
        shapefile = gpd.read_file(vector_file)
        X=[]
        y=[]
        cont=1
        total=len(shapefile)
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
                    re=re[re!=arr]
                    XTemp=np.reshape(rr,(int(np.shape(rr)[0]/shape[0]),shape[0]))

            for l in XTemp:
                if l not in X:                 
                    X.append(l)
                    y.append(shapefile['id'][i])
            print(f"Loop {cont} of {total} done!")
            cont+=1
        if trainTest==0:
            return X,y
        elif trainTest>=1:
            raise Exception("Error: you need to pass a value between 0 and 1.")
        else:
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=trainTest)
            # return X_train, X_test, y_train, y_test
            return train_test_split(X, y, test_size=trainTest)

        