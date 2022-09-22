from django.shortcuts import render

from .models import mlModels

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from tensorflow import keras

# Create your views here.

def home(request):
    return render(request, 'home.html')

def xception(request):
    petClassifierFiles = mlModels.objects.filter(priority=1)[0]
    path_arch = petClassifierFiles.architecture.path
    path_weights = petClassifierFiles.weights.path
    
    with open(path_arch) as json_file:
        json_config = json_file.read()
    
    model = tf.keras.models.model_from_json(json_config)
    model.load_weights(path_weights)
    
    if request.method == 'POST':
        handle_uploaded_file(request.FILES['sentFile'])
        
        image = tf.keras.preprocessing.image.load_img('static/test.jpg', target_size=(150,150,3))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) # Convert single image to a batch
        
        pred = tf.keras.activations.sigmoid(model.predict(input_arr))[0][0]
        
        
        caption = f'Perro con una probabilidad de: {"{:.4f}".format(pred*100)}%\n | gato con una probabilidad de: {"{:.4f}".format((1-pred)*100)}%'
        
        return render(request, 'xception.html', {'caption': caption})
    else:
        return render(request, 'xception.html')


def efficientNetB2(request):
    petClassifierFiles = mlModels.objects.filter(priority=2)[0]
    path_arch = petClassifierFiles.architecture.path
    path_weights = petClassifierFiles.weights.path
    
    with open(path_arch) as json_file:
        json_config = json_file.read()
    
    model = tf.keras.models.model_from_json(json_config)
    model.load_weights(path_weights)
    
    if request.method == 'POST':
        handle_uploaded_file(request.FILES['sentFile'])
        
        image = tf.keras.preprocessing.image.load_img('static/test.jpg', target_size=(150,150,3))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) # Convert single image to a batch
        
        pred = tf.keras.activations.sigmoid(model.predict(input_arr))[0][0]
        
        caption = f'Perro con una probabilidad de: {"{:.4f}".format(pred*100)}% | \ngato con una probabilidad de: {"{:.4f}".format((1-pred)*100)}%'
        
        return render(request, 'efficientNetB2.html', {'caption': caption})
    else:
        return render(request, 'efficientNetB2.html')
    
def handle_uploaded_file(f):
    with open('static/test.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)