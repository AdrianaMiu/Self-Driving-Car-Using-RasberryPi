from copy import copy
import os
import argparse
import cv2
import numpy as np
import sys
from time import sleep
from threading import Thread
import importlib.util
from VideoStream import *


#moatoare, senzorul de proximitate si senzorul cu infrarosu
from motoare import *
from senzor_HCSR04 import *
from follow_line import *

#senzor detectie linie
# follow()

# instantierea obiectului corespunzator senzorului de pe masinuta
hcsr = HcSr04(4, 27)


# Definirea si parsarea argumentelor de intrare
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='folderul in care este localizat fisierul .tflite',
                    default='traffic')
parser.add_argument('--graph', help='numele fisierului tflite daca este diferit de traffic.tflite',
                    default='traffic.tflite')
parser.add_argument('--labels',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='pragul minim de precizie in detectia obiectelor',
                    default=0.6)
parser.add_argument('--resolution', help='rezolutia camerei',
                    default='1280x720')
parser.add_argument('--edgetpu', help='utilizarea stickului pentru a accelera detectia',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Importarea librariilor Tensorflow --> pentru edgtpu: import  load_delegate 
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
# else:
#     from tensorflow.lite.python.interpreter import Interpreter
#     from tensorflow.lite.python.interpreter import load_delegate

# asignarea fisierului tflite compilat pentru edge tpu
if (GRAPH_NAME == 'traffic.tflite'):
        GRAPH_NAME = 'traffic14.tflite'       

# Calea directorului curent
CWD_PATH = os.getcwd()

# Calea catre fisierul .tflite, care contine modelul care este utilizat pentru detectia de obiecte
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Calea catre fisierul labelmap cu denumirile claselor
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Incarcarea labelmap-ului
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# # Incarcarea modelului tflite 
# #Pentru rularea cu ajutorul stickului se foloseste load_delegate argument
interpreter = Interpreter(model_path=PATH_TO_CKPT,
                        experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
print(PATH_TO_CKPT)
interpreter.allocate_tensors()

# Detalii despre model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Verfificarea stratului de iesire daca modelul este obtinut cu TF1 sau TF2
# deorece variabilele de iesire sunt ordonate diferit pentru TF1 si TF2
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # pentru TF2
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # pentru TF1
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# se initializeaza calculu frame rate-ului
frame_rate_calc = 1
freq = cv2.getTickFrequency() #returneaza in secunde frecventa semnalului de ceas

# Initializarea videostreamului
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

while True:
    
    # activarea senzorului
    result, v = hcsr.read()
    print("distance: {0} [cm]".format(v/10))

    

    # Pornirea timer-ului (pt calculul fps)
    t1 = cv2.getTickCount()# ne indica numarul de semnale de ceas care a fost trimis de la ev de ref.


    # Obtinerea frame-urilor de la videostream
    frame1=cv2.rotate(videostream.read(), cv2.ROTATE_180)
    # frame1=cv2.flip(frame1,1)
    print('frame')

    # colectarea frame-urilor si redimensionarea acestora[1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #color
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)# datele de intrare ale modelului

    # normarea pixelilor pentru un model necuantizat cu date de tip float
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # rularea detectiei--> modelul primeste ca data de intrare frameurile
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # rezultatele detectiei
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # cadranele pentru semnelele detectate 
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # indexul clasei pt obiectul detectat
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] #precizia detectiei

    # parcurgerea tuturor detectiilor si afisarea cadranelor daca detectia se afla sub un anumit scor minim de precizie impus
    object=[]
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # atasarea cadranelor
            # interpretorul poate returna coordonate care sunt in afara dimensiunilor imaginii, astfel se forteaza mentinerea acestora in imagine, utilizand min si max
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Adaugarea etichetelor de clasa
            object_name = labels[int(classes[i])] # cauta numele obiectlui din vectorul "labels" indexul clasei
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Exemplu: 'stop: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2) # dimensiunea fontului
            label_ymin = max(ymin, labelSize[1] + 10) # pt a ma asigura ca nu este adaugata eticheta foarte aproape de linia cadralului
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # o caseta alba in care sa fie afisata eticheta
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2) # 
            object=object_name
            print(object_name)
    # print(type(object_name))

            #coordonate
            # Draw circle in center
            # xcenter = xmin + (int(round((xmax - xmin) / 2)))
            # ycenter = ymin + (int(round((ymax - ymin) / 2)))
            # cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)


    # Calculul ratei de frame-uri
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    
    # Afisarea ratei de frame-uri un coltul imaginii
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    

    # afisarea frame-ului
    cv2.imshow('Object detector', frame)

    # initializarea miscarii robotului
    follow()

    #### CONTROUL MOTOARELOR ########
    if  v/10 <= 20.0 and v!=0.0:
        if object == 'stop':
            print("stop")
            stop()
        elif object == 'turn right':
            print("turn right")
            right()
            sleep(4)
            forward()
        elif object == 'straight ahead':
            print("straight ahead")
            forward()
        elif object == 'limit60km':
            print("limit60km")
            limit()
    else:
        follow()


    # gpio.cleanup()
    # inchiderea ferestrei prin apasarea tastei 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()
