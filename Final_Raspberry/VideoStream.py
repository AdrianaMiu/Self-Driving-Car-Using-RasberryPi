import cv2
import numpy as np
import sys
import time
from threading import Thread



class VideoStream:
    """Clasa corespondenta obiectului camera pe baza caruai se face streming Video de la camera Raspberry"""
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initizalizarea camerei si configurarea acesteia
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Citirea primului frame al stream-ului
        (self.grabbed, self.frame) = self.stream.read()

	# Variabila de contorol in cazul in care camera se opreste
        self.stopped = False

    def start(self):
	# Inceperea firului de executie care preia frame-urile streamului
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Mentinerea buclei pe timpul streamingului pana cand camera se opreste
        while True:
            # Daca camera este oprita, se inchide firul de executie
            if self.stopped:
                # Se inchid resursele camerei
                self.stream.release()
                return

            # In caz contrar, se trece la urmatorul frame 
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Returnaza cel mai recent frame
        return self.frame

    def stop(self):
	# Indica daca camera sau excutia ar trebui oprite
        self.stopped = True
