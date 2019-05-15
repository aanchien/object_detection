# object_detection

install python3 <br />
<br />
pip3 install tensorflow <br />
pip3 install flask <br />
pip3 install matplotlib <br />
pip3 install Pillow <br />
python3 detect.py <br />
<br />
Currently the model is trained for detecting barcode, pdf417 and qr code using a very small sample size. <br />
<br />
barcode_inference_graph was trained using faster_rcnn_inception_v2_coco_2018_01_28 <br />
barcode_inference_graph1 was trained using ssd_mobilenet_v1_coco<br />
 <br />
 Currently hardcoded to use barcode_inference_graph <br />
 
 # api
 
 curl -X POST \ <br />
  http://127.0.0.1/imageclassifier/predict/ \ <br />
  -H 'Accept: application/json' \ <br />
  -H 'Cache-Control: no-cache' \ <br />
  -H 'Content-Type: application/json' \ <br />
  -H 'cache-control: no-cache,no-cache' \ <br />
  -d '{  <br />
	"b64" : "" <br />
}' <br />
 <br />
 b64 takes a base64 decoded image in jpeg or png format.  <br />
  <br />
  The response contains the coordinates, the detected image type and the accurary from the point of view of the model<br />

