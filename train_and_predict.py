
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from IPython.display import clear_output
import numpy as np
import keras
import cv2
from PIL import Image
from keras.preprocessing import image


train_dir="D:\\dataset\\captcha\\train"
validation_dir="D:\\dataset\\captcha\\validation"



def Model_Resault(list_history):
    acc = (list_history.history['acc'])[len(list_history.history['acc'])-1]
    val_acc = (list_history.history['val_acc'])[len(list_history.history['val_acc'])-1]
    loss = (list_history.history['loss'])[len(list_history.history['loss'])-1]
    val_loss = (list_history.history['val_loss'])[len(list_history.history['val_loss'])-1]

    print('\n train ==> loss: %.3f,  acc: %.3f%%'%(loss, acc))
    print('\n validation ==> loss: %.3f,  acc: %.3f%%'%(val_loss, val_acc))

    
    
class Plot(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
        plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.i += 1
        
                
        clear_output(wait=True)
        plt.figure(num=2, figsize=(10, 5), dpi=120 )
        plt.subplot(1,2,1)
        plt.plot(np.array(self.x)+1, self.acc, label="acc",color="red")
        plt.plot(np.array(self.x)+1, self.val_acc, label="val_acc",color="green")
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.title('accuracy plot')
        plt.grid()
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(np.array(self.x)+1, self.loss,"-" ,label="loss",color="red")
        plt.plot(np.array(self.x)+1,self.val_loss, label="val_loss",color="green")
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title('loss plot')
        plt.grid()
        plt.legend()
        plt.show();

plot = Plot()


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',padding="same",
                        input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu',padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu',padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])



batch_Size=32



train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 32x32
        target_size=(32, 32),
        batch_size=batch_Size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(32, 32),
        batch_size=batch_Size,
        class_mode='categorical')


# In[ ]:


def file_count(folder):
    file_count=0
    for i in folder:
        path, dirs, files = next(os.walk(i))  
        file_count+= len(files)
    return file_count


# In[ ]:


fnames_train = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir)]
fnames_valid = [os.path.join(validation_dir, fname) for fname in os.listdir(validation_dir)]


history = model.fit_generator(
      train_generator,
      steps_per_epoch=int(file_count(fnames_train)/batch_Size),
      epochs=30,
      validation_data=validation_generator,
      validation_steps=int(file_count(fnames_valid)/batch_Size),callbacks=[plot],verbose=1,shuffle=True)

Model_Resault(history)


# In[ ]:


Model_Resault(history)

# model.save("captcha.h5")
# model=models.load_model("captcha.h5")


# In[ ]:



def crop(img):
    
    j=0
    b=1
    list_1=[]
    for i in range(45,190,45):
        #print(j," ",i)
        crop_img = img[:, j:i]
        #print(crop_img.shape)
        list_1.append(crop_img)
        j+=45
        plt.subplot(1,4,b)
#         plt.title(name[b-1])
        plt.imshow(crop_img,cmap = plt.get_cmap('gray'))
        b+=1
    plt.show()
    return list_1





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


# In[ ]:


predict='D:\\dataset\\captcha\\predict\\'
if not os.path.exists(predict):
        os.makedirs(predict)
        
def peredict(imagename):
    data = np.array(plt.imread(imagename))
    plt.imshow(data)
    plt.show()
    listNUM=crop(process_1(imagename))


    for j in range(4):
                I8 = (((listNUM[j] - listNUM[j].min()) / (listNUM[j].max() - listNUM[j].min())) * 255.9).astype(np.uint8)
                img = Image.fromarray(I8)
                img.save(predict+str(j)+".jpg")

    list_=[]
    for f in range(4):

        img = image.load_img(predict+str(f)+".jpg", target_size=(32, 32))

        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)
        list_.append(x)
    numlist=["1","2","3","4","6","7","8","9"]
    print("\n\t",end="")
    for f in range(4):

        print(" "+numlist[model.predict_classes(list_[f])[0]],end="")


# In[ ]:


imagename="C:\\Users\\ehsan\\Desktop\\rrrrr.jpg"
peredict(imagename)


# In[ ]:


fnames = [os.path.join("D:\\dataset\\captcha\\dataset", fname) for fname in os.listdir("D:\\dataset\\captcha\\dataset")]
for i in fnames[0:2]:
    peredict(i)

