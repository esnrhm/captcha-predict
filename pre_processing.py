
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import glob
import os
import os.path
from pathlib import Path
import random



captcha_image_folder = "D:\\dataset\\captcha\\dataset\\"
train_folder='D:\\dataset\\captcha\\train\\'


def crop(img,name):
    j=0
    b=1
    list_1=[]
    for i in range(45,190,45):
        #print(j," ",i)
       
        crop_img = img[:, j:i]
        #print(crop_img.shape)
        list_1.append(crop_img)
        j+=45
#         plt.subplot(1,4,b)
#         plt.title(name[b-1])
        
#         plt.imshow(crop_img,cmap = plt.get_cmap('gray'))
        b+=1
#     plt.show()
    return list_1
# Load the image


def process_1(image):
    data = np.array(plt.imread(image))

    im_gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    im_gray=(np.where(im_gray < 100, im_gray, 255))
    I8 = (((im_gray - im_gray.min()) / (im_gray.max() - im_gray.min())) * 255.9).astype(np.uint8)
    img = Image.fromarray(I8)
    img.save("file.png")
    img = cv2.imread('file.png', cv2.IMREAD_COLOR)
    ksize = (3,3) 
    img = cv2.blur(img, ksize)  
    blur = cv2.fastNlMeansDenoisingColored(img,None,26,10,7,21)
    I8 = (((blur - blur.min()) / (blur.max() - blur.min())) * 255.9).astype(np.uint8)
    img = Image.fromarray(I8)
    img.save("file.png")
    im_gray = cv2.imread('file.png', cv2.IMREAD_GRAYSCALE)
    im_gray=(np.where(im_gray < 100, im_gray, 255))
    return im_gray






counts={}
k=0 #loading counter


captcha_image_files = glob.glob(os.path.join(captcha_image_folder, "*"))
darsad=int(len(captcha_image_files))-1
k=0
for (i, captcha_image_file) in enumerate(captcha_image_files):
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    list_name=[]
    for letter_text in (captcha_correct_text):
        list_name.append(letter_text)
    listNUM=crop(process_1(captcha_image_folder+captcha_correct_text+".jpg"),list_name)
    #print(listNUM)
    for j in range(4):
            I8 = (((listNUM[j] - listNUM[j].min()) / (listNUM[j].max() - listNUM[j].min())) * 255.9).astype(np.uint8)
            img = Image.fromarray(I8)
            count = counts.get(list_name[j], 1) 
            counts[list_name[j]] = count + 1
            
            if not os.path.exists(train_folder+str(list_name[j])):
                os.makedirs(train_folder+str(list_name[j]))
            img.save(os.path.join(train_folder, str(list_name[j]),str(count).zfill(6)+".jpg"))
    k=int((i*100/darsad)/2)
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("\t[%-50s] %d%%" % ('='*k, k*2))
    sys.stdout.flush()
    
print("\n\n\tFINISH")
    




fnames = [os.path.join(train_folder, fname) for fname in os.listdir(train_folder)]
for i in fnames:
    if not os.path.exists(i.replace("train","validation")):
        os.makedirs(i.replace("train","validation"))

for i in fnames:
    captcha_image_files = glob.glob(os.path.join(i, "*"))
    for j in np.random.choice(range(0,len(captcha_image_files)), int(len(captcha_image_files)/5), replace=False):
        os.rename(captcha_image_files[j], captcha_image_files[j].replace("train","validation"))







