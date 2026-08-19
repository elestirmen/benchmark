import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from osgeo import gdal
import random

def dosyaya_yaz(model_name,epochs,sonuclar_dogru1,sonuclar_yanlis1,sonuclar_dogru2,sonuclar_yanlis2):    
    
    model_name="sonuclar_"+model_name
    sonuclar_dosya = open(model_name+".txt", "w")
    ortalama=(sonuclar_dogru1+sonuclar_dogru2)/2
    sonuclar = np.vstack((epochs,sonuclar_dogru1, sonuclar_yanlis1,sonuclar_dogru2, sonuclar_yanlis2,ortalama)).T
    print(sonuclar)
    
    df = pd.DataFrame(sonuclar, columns = ['epochs','dogru_tahmin1','yanlis_tahmin1','dogru_tahmin2','yanlis_tahmin2','ortalama'])
    
    sonuclar_dosya.write(df.to_string())
    sonuclar_dosya.close()
    df.to_csv(model_name+".csv", index=False)




#spatial çözünürlük elde etme
camera_sensor_genislik=6 #mavic2zoom için 6  milimetre sensör genişliği
camera_focal_lenght=4 #mavic2zoom için 4 milimetre
ucus_yuksekligi=75  #metre olarak uçuş yüksekliği
goruntu_piksel_genisligi = 4000 #pipksel olarak resmin genişliği
goruntu_piksel_yuksekligi = 3000 #pipksel olarak resmin genişliği
mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak



DATADIR_anlik_haritalar1 = r"parcalar\swistopo"
DATADIR_anlik_haritalar2 = r"parcalar\bingmap"

DATADIR_ana_haritalar = r"haritalar"

#from natsort import natsorted   #dosyaları doğru sıralamak için eklendi

anlik_klasoru1 =os.listdir(DATADIR_anlik_haritalar1)
anlik_klasoru2 =os.listdir(DATADIR_anlik_haritalar2)

haritalar_klasoru=os.listdir(DATADIR_ana_haritalar)


sonuclar_dogru1 = np.array([])
sonuclar_dogru2 = np.array([])
sonuclar_yanlis1 = np.array([])
sonuclar_yanlis2 = np.array([])
epochs = np.array([])

for i in range(len(haritalar_klasoru)):
    
    ana_harita =cv2.imread(str("haritalar/"+haritalar_klasoru[i]),0)  
    anlik_goruntu1 =cv2.imread(str("parcalar/swistopo/"+anlik_klasoru1[i]),0)
    anlik_goruntu2 =cv2.imread(str("parcalar/bingmap/"+anlik_klasoru2[i]),0)
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
    konum_dogru1=0
    konum_dogru2=0
    konum_yanlis1=0
    konum_yanlis2=0
    
    j=0
    while(True):
        
        template1=anlik_goruntu1[dikey-512:dikey,yatay-512:yatay]
        template2=anlik_goruntu2[dikey-512:dikey,yatay-512:yatay]
        #plt.imshow(template, cmap = "gray")   
        
        print(template1.shape)
        h,w =template1.shape
        
            
        res1= cv2.matchTemplate(ana_harita, template1, cv2.TM_CCOEFF, None)
        res2= cv2.matchTemplate(ana_harita, template2, cv2.TM_CCOEFF, None)
            
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
            
        print("swistopo_konum: ",max_val1, max_loc1)
        print("bing_konum: ",max_val2, max_loc2)

            
            
        top_left1 = max_loc1
        top_left2 = max_loc2
                    
        bottom_right1 = (top_left1[0] + w,top_left1[1] +h)
        bottom_right2 = (top_left2[0] + w,top_left2[1] +h)
        # ana_harita = cv2.cvtColor(ana_harita, cv2.COLOR_GRAY2BGR)
            
        # cv2.rectangle(ana_harita, top_left, bottom_right,(255,0,0),35)
        #     #plt.figure()
            
        #     # plt.imshow(img)
        #     # plt.title("Tespit edilen Sonuç"), plt.axis("on")
        #     # plt.suptitle(meth)
        #     # plt.pause(0.0001)
        # res = cv2.resize(ana_harita, dsize=(1000,1000), interpolation=cv2.INTER_CUBIC)
        # cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
        # cv2.imshow("Resized", res)
        # cv2.waitKey(100)
        #     #cv2.destroyAllWindows()
           
        # ana_harita = ana_harita_temp
        
        
        sira = j
        kenarx=int(ana_harita.shape[0]/512)
        kx = (sira % kenarx)*512
        ky = (int(sira/kenarx))*512
        
        
        j+=1
        print("sıra ",j)
        konum=""
        if abs(kx-top_left1[0]) < 512 and abs(ky-top_left1[1])<512:
            print("konum dogru")
            konum="dogru"
            konum_dogru1+=1 
        else:
            print("yanlis konum")
            konum_yanlis1+=1
            konum="yanlis"
            
        if abs(kx-top_left2[0]) < 512 and abs(ky-top_left2[1])<512:
            print("konum dogru")
            konum="dogru"
            konum_dogru2+=1 
        else:
            print("yanlis konum")
            konum_yanlis2+=1
            konum="yanlis"
            
            
        if yatay<anlik_goruntu1.shape[1]:
            yatay+=512
        else:
            yatay=512
            dikey+=512
        if dikey>anlik_goruntu1.shape[0]:
            break
         
       
        
    
    anlik_name="_"
    
    sonuclar_dogru1 = np.append(sonuclar_dogru1, konum_dogru1)
    sonuclar_yanlis1 = np.append(sonuclar_yanlis1, konum_yanlis1)
    sonuclar_dogru2 = np.append(sonuclar_dogru2, konum_dogru2)
    sonuclar_yanlis2 = np.append(sonuclar_yanlis2, konum_yanlis2)
    
    epochs = np.append(epochs,(i+1)*100)
    dosyaya_yaz(anlik_name,epochs,sonuclar_dogru1,sonuclar_yanlis1,sonuclar_dogru2,sonuclar_yanlis2)
                
            
            
            
        
        
    
        