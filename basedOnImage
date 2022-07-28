#all majour import commands
import os
import warnings
import math
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
import keras
from google.colab import drive
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAvgPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

warnings.filterwarnings('ignore')
drive.mount('/content/drive/')

#link your google drive to google cloab and mount the data set here
rootDirectory = '/content/drive/MyDrive/AIAndML/LiverDataset'
imageCount={}
for i in os.listdir(rootDirectory):
  imageCount[i]=len(os.listdir(os.path.join(rootDirectory,i)))
imageCount.items()

#here data set is being splitted into different folders 
#Uncomment os.remove(original) if you want to remove original images from google drive
def dataFolder(path,split):
  if not os.path.exists("./"+path):
    os.mkdir("./"+path)
    for i in os.listdir(rootDirectory):
      os.makedirs("./"+path+"/"+i)
      for j in np.random.choice(a = os.listdir(os.path.join(rootDirectory,i)),
                                size = (math.floor(split*imageCount[i]-5)),
                                replace = False ):
        original = os.path.join(rootDirectory,i,j)
        duplicate = os.path.join("./"+path,i)
        shutil.copy(original,duplicate)
        #os.remove(original)
  else :
    print(f'{path} already exists! \n')

#Enter desired data split sizes
dataFolder('train',0.7)
dataFolder('validation',0.7)
dataFolder('test',0.7)

#this function will return a dictionary having ('folder name', numberOfImagesPresent)
def folderSize(path):
  size={}
  for i in os.listdir(path):
    size[i]=len(os.listdir(os.path.join(path,i)))
  print(f'items under {path} are : ', size.items())

folderSize('/content/test/')
folderSize('/content/train/')
folderSize('/content/validation/')

"""CNN Model """

model = Sequential()

model.add(Conv2D (filters = 16, kernel_size = (3,3), activation = 'relu', input_shape = (224,224,3) ))

model.add(Conv2D (filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (224,224,3) ))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D (filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = (224,224,3) ))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D (filters = 128, kernel_size = (3,3), activation = 'relu', input_shape = (224,224,3) ))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(rate = 0.25))

model.add(Flatten())

model.add(Dense(units = 64, activation ='relu'))
model.add(Dropout(rate = 0.25))
model.add(Dense(units = 1, activation ='sigmoid'))

model.summary()

model.compile(optimizer = 'adam',loss = keras.losses.binary_crossentropy, metrics = ['accuracy'] )

"""Data generation process 💽"""

def imagePreprocessor(path):
  imageData = ImageDataGenerator (zoom_range = 0.2, shear_range = 0.2,  rescale = 1/255, horizontal_flip = True)
  image = imageData.flow_from_directory(directory = path, target_size = (224,224), batch_size = 32, class_mode = 'binary')
  return image

path = '/content/train'
trainData = imagePreprocessor(path)

def imagePreprocessor1(path):
  imageData = ImageDataGenerator (rescale = 1/255)
  image = imageData.flow_from_directory(directory = path, target_size = (224,224), batch_size = 32, class_mode = 'binary')
  return image

path = '/content/test'
testData = imagePreprocessor1(path)

path = '/content/validation'
validationData = imagePreprocessor1(path)

#early stopping and model check point
es = EarlyStopping ( monitor = 'val_accuracy', min_delta = 0.01, patience = 5, verbose =1, mode = 'auto') 
mc = ModelCheckpoint ( monitor = 'val_accuracy', filepath = './bestModel.h5', verbose = 1, save_best_only = True, mode = 'auto')
cd = [es, mc]

"""Training model"""

hst = model.fit_generator( generator = trainData, steps_per_epoch = 8, epochs = 5, verbose = 1,
                              validation_data = (validationData), validation_steps= 16, callbacks = cd )

#graphical interpretation 
h = hst.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c = 'red')
plt.title ('accuracy vs validation accuracy')
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'], c = 'red')
plt.title ('accuracy vs validation loss')
plt.show()

model = load_model('/content/bestModel.h5')

accuracy = model.evaluate_generator(testData)[1]
print (f'Model accuracy is {accuracy} ')
