import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import matplotlib.pyplot as plt
from osgeo import gdal


#spatial çözünürlük elde etme
camera_sensor_genislik=6 #mavic2zoom için 6  milimetre sensör genişliği
camera_focal_lenght=4 #mavic2zoom için 4 milimetre
ucus_yuksekligi=75  #metre olarak uçuş yüksekliği
goruntu_piksel_genisligi = 4000 #pipksel olarak resmin genişliği
goruntu_piksel_yuksekligi = 3000 #pipksel olarak resmin genişliği
mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak



anlik_goruntu="anlik112.jpg"

#template matching
harita="harita_swistopo.jpg"
#harita="harita_gmap.jpg"

img = cv2.imread(harita,0)
print(img.shape)

kenarx=int(img.shape[0]/512)


kx = (112 % kenarx)*512
ky = (int(112/kenarx))*512




# gdal.Warp('anlik_goruntu_warped.tif', anlik_goruntu, xRes=0.09, yRes=0.09) 
# raster = gdal.Open('anlik_goruntu_warped.tif')
# gt =raster.GetGeoTransform()

# print (gt)
# pixelSizeX = gt[1]
# pixelSizeY = -gt[5]
# print ("x = ",pixelSizeX)
# print ("y = ",pixelSizeY)


template= cv2.imread(anlik_goruntu,0)
plt.imshow(template, cmap = "gray")

print(template.shape)
h,w =template.shape


#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

methods =['cv2.TM_CCOEFF']
for meth in methods:
    method  = eval(meth)    #stringleri fonksiyona çeviren fonksiyona
    res= cv2.matchTemplate(img, template, method, None, template)
    print(res.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    print(min_val, max_val, min_loc, max_loc)
    
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left =min_loc
    else:
        top_left = max_loc
            
    bottom_right = (top_left[0] + w,top_left[1] +h)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.rectangle(img, top_left, bottom_right,(255,0,0),35)
    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap = "gray")
    plt.title("Eşleşen Sonuç"), plt.axis("on")
    plt.subplot(122), plt.imshow(img)
    plt.title("Tespit edilen Sonuç"), plt.axis("on")
    plt.suptitle(meth)
    img = cv2.imread(harita,0)
    
dogru_konum=0
yanlis_konum=0
if abs(kx-top_left[0]) < 512 and abs(ky-top_left[1]<512):
    print("konum dogru")
    dogru_konum+=1 
else:
    print("yanlis kounm")
    yanlis_konum+=1
    
    