from scripts import readimg,segmentation,classification
import os
import numpy as np
import time

if __name__=='__main__':
    # try:
    #     im=r'samples/Ortomosaico.tif'
    #     img=readimg.imread(im)
    #     s=segmentation.Seg(img)
    #     s.SLIC(100,1.0,1.0,os.path.join(r'samples/output','SLIC.tif'))
    # except MemoryError as e:
    #     print(e)

    # print("END!")
    # while True:
    #     pass
    im=r'samples/Seg_FalsaCorTGI.tif'
    img=readimg.imread(im)
    shp=r'samples/AmostrasTGI2.shp'
    imOut=r'samples/output/Classified.tif'

    t1=time.time()
    c=classification.classify(img)
    X, X_test, y, y_test = c.get_samples(shp,column='Classvalue',trainTest=0.25)
    r=c.randomForest(X,y,imOut,X_test=X_test,y_test=y_test)
    t2=time.time()
    print(f"Classification finished in {t2-t1} seconds with accuracy of {r}")




