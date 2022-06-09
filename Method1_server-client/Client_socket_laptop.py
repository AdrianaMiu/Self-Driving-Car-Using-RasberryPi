import cv2 
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
import time
from imutils.video import FPS
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from operator import itemgetter

import socket, imutils
import time
import base64

BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET,
                                socket.SOCK_DGRAM) 
#foloseste UDP(SOCK_DGRAM) sau TCP (SOCK_STREAM) 
client_socket.setsockopt(socket.SOL_SOCKET,
                         socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = '172.20.10.14' 
print(host_ip)
port = 9999
message = b'Hello'

client_socket.sendto(message,(host_ip,port))
fps,st,frames_to_count,cnt = (0,0,20,0)

CUSTOM_MODEL_NAME = 'my_ssd_mobnet_tuned_25++' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
     'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
     'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

labels = [{'name':'stop', 'id':1}, {'name':'straight ahead', 'id':2}, {'name':'limit60km', 'id':3}]

"""             INCARCAREA MODELULUI DIN CHECKPOINT                 """
# Incarcarea fisierului de configurare (pipeline.config) si construirea modelului de detectie
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False) #is_training=False =>rularea modelului pe setul de validare

# Utilizarea checkpointului
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-20')).expect_partial()

@tf.function #funcția este utilă în crearea și utilizarea graficelor de calcul
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections 

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
 
while True:#cap.isOpened(): 
    # ret, frame = cap.read()
    packet,_ = client_socket.recvfrom(BUFF_SIZE)
    data = base64.b64decode(packet,' /')
    npdata = np.fromstring(data,dtype=np.uint8)
    frame = cv2.imdecode(npdata,1)
    frame = cv2.putText(frame,'FPS: '+str(fps),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    image_np = np.array(frame)
   
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    print("\n frame")
    # print(input_tensor)
    
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
                
    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
#Debug clasa si scor
    # print('id_clasa')
    # print(detections['detection_classes'])
    # print("dimensiune"+str(detections['detection_classes'].size))
    # print("scor")
    # print(type(detections['detection_scores']))
    # print((detections['detection_scores']).size)
    # print(detections['detection_scores'][0])
    
    detected_sign=''
    if detections['detection_scores'].size != 0:
        if detections['detection_scores'][0] >= 0.8:
            print(detections['detection_classes'])
            id=int(detections['detection_classes'][0])+1
            detected_sign=category_index[id]['name']
    
    if cnt == frames_to_count:
        try:
            fps = round(frames_to_count/(time.time()-st))
            st=time.time()
            cnt=0
        except:
            pass
    cnt+=1     
    msg=bytes(detected_sign.encode()) # convertire din str in binar
    client_socket.sendto(msg,(host_ip,port))
    print(msg.decode('utf-8'))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        # cap.release()
        client_socket.close()
        cv2.destroyAllWindows()
        break
 