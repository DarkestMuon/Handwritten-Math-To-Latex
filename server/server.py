from flask import Flask, request, send_from_directory
from segmentation import *
from evalulator import *
from classifier import *

app = Flask(__name__)

@app.route('/demo/prediction', methods=['get','post'])
def prediction():
  
  # deletes previous images 
  removing_files = glob.glob('../images/*.png')
  for i in removing_files:
      os.remove(i)

  image = request.files['image']
  path = "../images/" + image.filename
  image.save(path)

  segmentation(path)

  x = parser(predict())

  return x


if __name__ == '__main__':
  app.run(port=8000, debug=True)