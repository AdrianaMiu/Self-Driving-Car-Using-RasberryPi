from time import sleep
import RPi.GPIO as gpio


gpio.setmode(gpio.BCM)
gpio.setwarnings(False)

SL=10
SR=9
left_ir=gpio.setup(SL,gpio.IN) #GPIO 2 -> Left IR out
right_ir=gpio.setup(SR,gpio.IN) #GPIO 3 -> Right IR out 

EnaA=12
In1A=17
In2A=22
EnaB=13
In1B=24
In2B=23
gpio.setup(EnaA,gpio.OUT)
gpio.setup(In1A,gpio.OUT)
gpio.setup(In2A,gpio.OUT)
gpio.setup(EnaB,gpio.OUT)
gpio.setup(In1B,gpio.OUT)
gpio.setup(In2B,gpio.OUT)
pwmA = gpio.PWM(EnaA, 10)
pwmA.start(0)
pwmB = gpio.PWM(EnaB, 10)
pwmB.start(0)
 
def move(speedL,speedR): # cele mai bune rez pt forward le-am obtinut pt L=0.725 si R=1
 
        pwmA.ChangeDutyCycle(abs(speedR))
        pwmB.ChangeDutyCycle(abs(speedL))
 
        if speedL>0:
            gpio.output(In1A,gpio.HIGH)
            gpio.output(In2A,gpio.LOW)
        else:
            gpio.output(In1A,gpio.LOW)
            gpio.output(In2A,gpio.HIGH)
 
        if speedR>0:
            gpio.output(In1B,gpio.HIGH)
            gpio.output(In2B,gpio.LOW)
        else:
            gpio.output(In1B,gpio.LOW)
            gpio.output(In2B,gpio.HIGH)

    
        

def stop(t):
        pwmA.ChangeDutyCycle(0)
        pwmB.ChangeDutyCycle(0)
        sleep(t)
        
 


def forward():
    move(16.5,16.5)

def right():
    move(0.65,0.2)

def left():
    move(0.2,0.7)

def limit():
    move(0.2,0.8)


try:
    while True:
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
            stop(5)
except KeyboardInterrupt:
        stop(1)
        gpio.cleanup()

# try:
#     while True:
#         forward()
#         # gpio.cleanup()
# except KeyboardInterrupt:
#     stop(1)
#     gpio.cleanup()