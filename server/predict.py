from segmentation import *
from evalulator import *
from classifier import *

predict("/content/img")
def predict(image_path):

  # creates ROI's
  segmentation(image_path)

  # returns latex string
  return parser(predict())

