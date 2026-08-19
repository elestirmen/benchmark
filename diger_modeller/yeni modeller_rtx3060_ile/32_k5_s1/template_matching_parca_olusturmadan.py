import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from osgeo import gdal
import random

def dosyaya_yaz(model_name,epochs,sonuclar_dogru,sonuclar_yanlis):    
    
    model_name="sonuclar_"+model_name
    sonuclar_dosya = open(model_name+".txt", "w")
    sonuclar = np.vstack((epochs,sonuclar_dogru, sonuclar_yanlis)).T
    print(sonuclar)
    
    df = pd.DataFrame(sonuclar, columns = ['epochs','dogru_tahmin','yanlis_tahmin'])
    
    sonuclar_dosya.write(str(df))
    
    df.to_csv(model_name+"csv", index=False)




#spatial çözünürlük elde etme
camera_sensor_genislik=6 #mavic2zoom için 6  milimetre sensör genişliği
camera_focal_lenght=4 #mavic2zoom için 4 milimetre
ucus_yuksekligi=75  #metre olarak uçuş yüksekliği
goruntu_piksel_genisligi = 4000 #pipksel olarak resmin genişliği
goruntu_piksel_yuksekligi = 3000 #pipksel olarak resmin genişliği
mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak



DATADIR_anlik_haritalar = r"parcalar"

DATADIR_ana_haritalar = r"haritalar"

from natsort import natsorted   #dosyaları doğru sıralamak için eklendi

anlik_klasoru =os.listdir(DATADIR_anlik_haritalar)

haritalar_klasoru=os.listdir(DATADIR_ana_haritalar)


sonuclar_dogru = np.array([])
sonuclar_yanlis = np.array([])
epochs = np.array([])

for i in range(len(haritalar_klasoru)):
    anlik_goruntu =cv2.imread(str("parcalar/"+anlik_klasoru[i]),0)
    ana_harita =cv2.imread(str("haritalar/"+haritalar_klasoru[i]),0)  
    print(ana_harita.shape)



    # gdal.Warp('anlik_goruntu_warped.tif', anlik_goruntu, xRes=0.09, yRes=0.09) 
    # raster = gdal.Open('anlik_goruntu_warped.tif')
    # gt =raster.GetGeoTransform()
    
    # print (gt)
    # pixelSizeX = gt[1]
    # pixelSizeY = -gt[5]
    # print ("x = ",pixelSizeX)
    # print ("y = ",pixelSizeY)
    
    
    
    
    dikey=512
    yatay=512
    
    ana_harita_temp=ana_harita
    
    
    #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #methods =['cv2.TM_CCOEFF']
    konum_dogru=0
    konum_yanlis=0
    
    j=0
    while(True):
        
        template=anlik_goruntu[dikey-512:dikey,yatay-512:yatay]
        #plt.imshow(template, cmap = "gray")   
        
        print(template.shape)
        h,w =template.shape
        
            
        res= cv2.matchTemplate(ana_harita, template, cv2.TM_CCOEFF, None)
            
        print(res.shape)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
        print("konum: ",max_val, max_loc)
            
            
        top_left = max_loc
                    
        bottom_right = (top_left[0] + w,top_left[1] +h)
        #ana_harita = cv2.cvtColor(ana_harita, cv2.COLOR_GRAY2BGR)
            
        cv2.rectangle(ana_harita, top_left, bottom_right,(255,0,0),35)
            #plt.figure()
            
            # plt.imshow(img)
            # plt.title("Tespit edilen Sonuç"), plt.axis("on")
            # plt.suptitle(meth)
            # plt.pause(0.0001)
        res = cv2.resize(ana_harita, dsize=(1000,1000), interpolation=cv2.INTER_CUBIC)
        cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
        cv2.imshow("Resized", res)
        cv2.waitKey(100)
            #cv2.destroyAllWindows()
           
        ana_harita = ana_harita_temp
        
        
        sira = j
        kenarx=int(ana_harita.shape[0]/512)
        kx = (sira % kenarx)*512
        ky = (int(sira/kenarx))*512
        
        
        j+=1
    
        konum=""
        if abs(kx-top_left[0]) < 512 and abs(ky-top_left[1])<512:
            print("konum dogru")
            konum="dogru"
            konum_dogru+=1 
        else:
            print("yanlis kounm")
            konum_yanlis+=1
            konum="yanlis"
            
            
        if yatay<anlik_goruntu.shape[1]:
            yatay+=512
        else:
            yatay=512
            dikey+=512
        if dikey>anlik_goruntu.shape[0]:
            break
    
    anlik_name="_"
    
    sonuclar_dogru = np.append(sonuclar_dogru, konum_dogru)
    sonuclar_yanlis = np.append(sonuclar_yanlis, konum_yanlis)
    epochs = np.append(epochs,(i+1)*100)
    dosyaya_yaz(anlik_name,epochs,sonuclar_dogru,sonuclar_yanlis) 
                
            
            
            
        
        
    
        