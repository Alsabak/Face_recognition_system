import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import cv2
from google.colab import files
import zipfile
import io
import pandas as pd


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))  
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

dir_name=(C:\Users\acer\Downloads\dataset)
y=[]; x=[]; target_names=[]
person_id=0; h=w=300
n=samples=0
class_names=[]
for person_name in os.listdir(dir_name):
    #print(person_name)
    dir_path=dir_name+person_name+"/"
    class_names.append(person_name)
    for image_name in os.listdir(dir_path):
      #formulate the image path
      image_path=dir_path+image_name
      #read the input image
      img=cv2.imread(image_path)
      #convert into grayscale
      gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      #resize image to 300*300 dimension
      resized_image=cv2.resize(gray,(h,w))
      #convert matrix to vector
      v= resized_image.flatten()
      x.append(v)
      #increase the number of samples
      n_samples= n_samples +1
      #adding the categorical label
      y.append(person_id)
      #adding the person name
      target_names.append(person_name)
      #increase the person id by 1
      person_id= person_id + 1
#transfer list to numpy array
y=np.array(y)
x=np.array(x)
target_names=np.array(target_names)
n_features=x.shape[1]
print(y.shape,x.shape,target_names.shape)
print("Number of samples: ", n_samples)
n_classes= target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
