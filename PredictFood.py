from __future__ import print_function
import boto3
import os
import sys
import uuid
from PIL import Image
import PIL.Image
import tensorflow as tf
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np    
     
s3_client = boto3.client('s3')
     
def __init__(self, modelpath, labels):
    self.model = load_model(modelpath)
    self.labels = labels
    
def predict(self,img):
    #check if image is a filepath
    if(isinstance(img,str)):
        if(not os.path.exists(img)):
            print("Error: Invalid File Path")
            return ""
        else:
            #if its a filepath, convert to PIL image
            img = Image.open(img)
        
    #resize image
    imgr = img.resize((128,128))
    x = img_to_array(imgr).reshape((1,128,128,3))
        
    #predict
    prediction = self.model.predict(x)
        
    #get max of predictions and return label(s)
    predIdx = np.argmax(prediction[0,:])
    return self.labels[predIdx]
     
def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key'] 
        download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
        upload_path = '/tmp/classified-{}'.format(key)
        
        #check if model is downloaded
        #change path to be "local"
        modelpath = '/tmp/cnnModelDEp80.h5'
        if(isinstance(modelpath,str)):
            if(not os.path.exists(modelpath)):
                modelpath = '/tmp/cnnModelDEp80.h5'
                s3_client.download_file(bucket, 'cnnModelDEp80.h5', modelpath)

        labels = ['apple', 'banana']
        __init__(modelpath, labels)
        
        s3_client.download_file(bucket, key, download_path)
        
        #call predict
        result = predict(download_path)
        
        #write result to txt
        txtpath = '/tmp/classified-{}.txt'.format(key)
        file = open(txtpath, 'w')
        file.write('{}', result[0])
        file.close()
        
        #upload back to s3
        s3_client.upload_file(txtpath, '{}'.format(bucket), 'classified-{}.txt'.format(key))
