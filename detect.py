import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import base64
import io
from flask import Flask, request, jsonify

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'barcode_inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

class Model:
   def __init__(self):
      self.graph = detection_graph
      self.session = tf.Session(graph=detection_graph)

   def predict(self, image):
      ops = self.graph.get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)
      image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
      output_dict = self.session.run(tensor_dict,
                             feed_dict={image_tensor: image})
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      return output_dict

m = Model()

app = Flask(__name__)

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'

@app.route('/imageclassifier/predict/', methods=['POST'])
def image_classifier():
   data = request.get_json()
   image = Image.open(io.BytesIO(base64.b64decode(data['b64'])))

   if not image.mode == 'RGB':
      image = image.convert('RGB')

   image_np = load_image_into_numpy_array(image)
   image_np_expanded = np.expand_dims(image_np, axis=0)
   output_dict = m.predict(image_np_expanded)

   vis_util.visualize_boxes_and_labels_on_image_array(
       image_np,
       output_dict['detection_boxes'],
       output_dict['detection_classes'],
       output_dict['detection_scores'],
       category_index,
       instance_masks=output_dict.get('detection_masks'),
       use_normalized_coordinates=True,
       line_thickness=8)
  
   lst = []
   for i in range(len(output_dict['detection_boxes'])) :
      if (int(output_dict['detection_scores'][i]*100))>49 :
         d = {}
         d['coordinates'] = ', '.join(str(v) for v in output_dict['detection_boxes'][i])
         d['class'] = int(output_dict['detection_classes'][i])
         d['accuracy'] = int(output_dict['detection_scores'][i]*100)
         lst.append(d)
   im = Image.fromarray(image_np)
   im.save("test.jpg")
   return json.dumps(lst)

if __name__ == '__main__':
   app.run(threaded=True)
 
