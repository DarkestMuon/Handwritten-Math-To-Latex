import tensorflow as tf
import numpy as np
from segmentation import *
import os
import glob

# load the keras model
model = tf.keras.models.load_model('../models')

class_names = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          '=', 'A', 'C', 'Delta', 'G', 'H', 'M', 'N', 'R', 'S', 'T', 'X', '[', ']', 'alpha', 
          'ascii_124', 'b', 'beta', 'cos', 'd', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 
          'gamma', 'geq', 'gt', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 
          'lim', 'log', 'lt', 'mu', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'rightarrow', 
          'sigma', 'sin', 'sqrt', 'sum', 'tan', 'theta', 'times', 'u', 'v', 'w', 'y', 'z', '{', '}']

def predict():

  folder_directory = '/content/img'
  symbols = []

  for image in sorted(os.listdir(folder_directory)):

    # load and resize the image to 45, 45
    image = tf.keras.preprocessing.image.load_img(folder_directory + '/' + image, 
                                                 target_size=(45,45), interpolation='nearest')

    # convert to array so we can feed them into our CNN
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    
    # shows the resized images - USED IN DEMO
    # cv2.imshow('', input_arr)
    # cv2.waitKey(0)

    # converts to numpy array
    input_arr = np.array([input_arr])

    # predicts using the CNN model
    predictions = model.predict(input_arr)
    class_name = class_names[np.argmax(predictions[0])]
    symbols.append(class_name)

  # deletes images once all predictions have been made
  removing_files = glob.glob('../rois/*.png')
  for i in removing_files:
      os.remove(i)

  # returns list of predictions
  return symbols
