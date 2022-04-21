from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from flask import jsonify
import threading
import time
import cv2
from textAreaDetector2 import LicencePlatesParser

plate_chars = "hurr"
output_frame = None
lock = threading.Lock()
app = Flask(__name__)
vs = VideoStream(src=1).start()
time.sleep(2.0) # for warm up
parser = LicencePlatesParser()

@app.route("/")
def index():
    return render_template("index.html", plate_chars=plate_chars)


def read_data():
    global vs, output_frame, lock
    while True:
        frame = vs.read()
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

@app.route("/detect")
def detection():
    global vs
    frame = vs.read()
    text_result, debug_image = parser.run(frame)
    if text_result:
        return jsonify({"detected":text_result})
    else:
        return jsonify({"detected":"None"})

def web_app(host, port):
    # textAreaDetector.loadNet()
    t = threading.Thread(target=read_data)
    t.daemon = True
    t.start()
    app.run(host=host, port=port, debug=True, threaded=True, use_reloader=False)


if __name__ == '__main__':
    web_app('0.0.0.0', '8080')

vs.stop()
