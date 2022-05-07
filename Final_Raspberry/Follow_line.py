from time import sleep
import RPi.GPIO as gpio
from motoare import *

gpio.setmode(gpio.BCM)
gpio.setwarnings(False)

SL=10
SR=9
left_ir=gpio.setup(SL,gpio.IN)
right_ir=gpio.setup(SR,gpio.IN)

#functie de urmarire linie
def follow():
        state_l=gpio.input(SL)
        state_r=gpio.input(SR)
        print ("right:", state_r)
        print("left:",state_l)
        if (state_r==False and state_l==False):
            forward()
        elif (state_r==True and state_l==False):
            right()
        elif (state_r==False and state_l==True):
            left()
        else:
            motor.stop(5)

####################TEST###########################


# while True:
#     follow()

