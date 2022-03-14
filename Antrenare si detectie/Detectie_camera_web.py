from __future__ import division
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

CUSTOM_MODEL_NAME = 'my_ssd_mobnet_tuned_10' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    # 'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
     'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    # 'APIMODEL_PATH': os.path.join('Tensorflow','models'),
     'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    # 'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    # 'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    # 'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    # 'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    # 'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    # 'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

labels = [{'name':'stop', 'id':1}, {'name':'straight ahead', 'id':2}, {'name':'limit60km', 'id':3}]
# import time

# _start_time = time.time()

# def tic():
#     global _start_time 
#     _start_time = time.time()

# def tac():
#     t_sec = time.time() - _start_time
   
#     print(' {}sec'.format(t_sec))

# def tac():
#     t_sec = time.time() - _start_time
   
#     print('{}'.format(t_sec))

"""             INCARCAREA MODELULUI DIN CHECKPOINT                 """
# Incarcarea fisierului de configurare (pipeline.config) si construirea modelului de detectie
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False) #is_training=False =>rularea modelului pe setul de validare

# Utilizarea checkpointului
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-31')).expect_partial()

@tf.function #funcția este utilă în crearea și utilizarea graficelor de calcul
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    print(type(detections))
    return detections 

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# time.sleep(2.0)
# fps = FPS().start()
while cap.isOpened(): 
    # tic()
    ret, frame = cap.read()
    image_np = np.array(frame)
    # print("\n timp achizitie imagine per frame:")
    # tac()
    # tic()
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    print("\n frame")
    # print(input_tensor)
    # print("\n timp procesare per frame:")
    # tac()
    # tic()
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # print("\n timp detectie per frame:")
    # tac()
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
                min_score_thresh=.9,
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
         if detections['detection_scores'][0] >= 0.9:
            id=int(detections['detection_classes'][0])+1
            detected_sign=category_index[id]['name']
    print(detected_sign)        
            
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
#     fps.update()
# # fps.stop()
# print("[INFO] timp trecut: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 