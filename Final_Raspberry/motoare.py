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
        self.pwmA = gpio.PWM(self.EnaA, 15)
        self.pwmA.start(0)
        self.pwmB = gpio.PWM(self.EnaB, 15)
        self.pwmB.start(0)
 
    def move(self,speedL,speedR): 
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
 
        
    def stop(self,t):
        self.pwmA.ChangeDutyCycle(0)
        self.pwmB.ChangeDutyCycle(0)
        sleep(t)
 
#pinii motoarelor
motor=Motor(12, 17, 22, 13, 23, 24) 

#evenimente in trafic
def forward():
    motor.move(17, 17)

def stop():
    motor.stop(5)

def right():
    motor.move(-10,10)

def left():
    motor.move(10, -10)    

def limit():
    motor.move(8,8)