from flask import Flask, render_template, Response 
import picamera 
import cv2
import socket 
import io 
from camera_pi import Camera

app = Flask(__name__) 
vc = cv2.VideoCapture(0) 
@app.route('/') 
def index(): 
   """Video streaming .""" 
   return render_template('index.html') 
def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed') 
def video_feed(): 
   """Video streaming route. Put this in the src attribute of an img tag.""" 
   return Response(gen(Camera()), 
                   mimetype='multipart/x-mixed-replace; boundary=frame') 
if __name__ == '__main__': 
	app.run(host='172.20.10.14' ,port=8000, debug=True, threaded=True) 