from time import sleep
from turtle import forward
import RPi.GPIO as gpio
gpio.setmode(gpio.BCM)
gpio.setwarnings(False)
 
class Motor():
    def __init__(self,EnaA,In1A,In2A,EnaB,In1B,In2B): # A este pt left si B pt right
        self.EnaA = EnaA
        self.In1A = In1A
        self.In2A = In2A
        self.EnaB = EnaB
        self.In1B = In1B
        self.In2B = In2B
        gpio.setup(self.EnaA,gpio.OUT)
        gpio.setup(self.In1A,gpio.OUT)
        gpio.setup(self.In2A,gpio.OUT)
        gpio.setup(self.EnaB,gpio.OUT)
        gpio.setup(self.In1B,gpio.OUT)
        gpio.setup(self.In2B,gpio.OUT)
        self.pwmA = gpio.PWM(self.EnaA, 100)
        self.pwmA.start(0)
        self.pwmB = gpio.PWM(self.EnaB, 100)
        self.pwmB.start(0)
 
    def move(self,speedL,speedR): # cele mai bune rez pt forward le-am obtinut pt L=0.725 si R=1
        speedL *=100
        speedR *=100
        if speedL>100: speedL=100
        elif speedL<-100: speedL= -100
        if speedR>100: speedR=100
        elif speedR<-100: speedR= -100
 
        self.pwmA.ChangeDutyCycle(abs(speedR))
        self.pwmB.ChangeDutyCycle(abs(speedL))
 
        if speedL>0:
            gpio.output(self.In1A,gpio.HIGH)
            gpio.output(self.In2A,gpio.LOW)
        else:
            gpio.output(self.In1A,gpio.LOW)
            gpio.output(self.In2A,gpio.HIGH)
 
        if speedR>0:
            gpio.output(self.In1B,gpio.LOW)
            gpio.output(self.In2B,gpio.HIGH)
        else:
            gpio.output(self.In1B,gpio.HIGH)
            gpio.output(self.In2B,gpio.LOW)
 
        
    def stop(self):
        self.pwmA.ChangeDutyCycle(0)
        self.pwmB.ChangeDutyCycle(0)
        
 
def forward():
    motor.move(0.7,0.7)

motor=Motor(12,17,22,13,23,24)


while True:
    forward()