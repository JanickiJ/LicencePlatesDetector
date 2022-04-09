import textAreaDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import time
import cv2

output_frame = None
lock = threading.Lock()
app = Flask(__name__)
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
    return render_template("index.html")

def read_data():
    global vs, output_frame, lock
    while True:
        frame = vs.read()
        # TODO : change below section for plate recognition, now it draws areas
        # where text is detected

        # section start
        boxes = textAreaDetector.textAreas(frame.copy())
        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # section end
        
        with lock:
            output_frame = frame.copy()


def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)

            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    textAreaDetector.loadNet()
    host = '0.0.0.0'
    port = '8080'
    t = threading.Thread(target=read_data)
    t.daemon = True
    t.start()
    app.run(host=host, port=port, debug=True, threaded=True, use_reloader=False)

vs.stop()