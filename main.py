from segmentation import *
from evalulator import *
from classifier import *

def main(image_path):

  # creates ROI's
  segmentation(image_path)

  # returns latex string
  return parser(predict())

# samples 1-4 - USED IN DEMO
# a = main('samples/sample 1.png')
# b = main('samples/sample 2.png')
# c = main('samples/sample 3.png')
# d = main('samples/sample 4.png')

# print('sample 1:',a)
# print('sample 2:',b)
# print('sample 3:',c)
# print('sample 4:',d)

