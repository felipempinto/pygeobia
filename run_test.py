from segment import readimg,segmentation
import os

if __name__=='__main__':
    try:
        im=r'samples/Ortomosaico.tif'
        img=readimg.imread(im)
        s=segmentation.Seg(img)
        s.SLIC(100,1.0,1.0,os.path.join(r'samples/output','SLIC.tif'))
    except MemoryError as e:
        print(e)

    print("END!")
    while True:
        pass


