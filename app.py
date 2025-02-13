from flask import Flask, render_template, request, Response, jsonify, url_for
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Global variable for webcam stream
camera_stream = None

# Human Detection Model (HOG + SVM)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

@app.route('/')
def index():
    return render_template('index.html')

# 🔹 Start Webcam Feed
def generate_frames():
    global camera_stream

    if camera_stream is None or not camera_stream.isOpened():
        camera_stream = cv2.VideoCapture(0)

    while True:
        success, frame = camera_stream.read()
        if not success:
            break

        # Human Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.1)

        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 🔹 Stop Webcam
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_stream
    if camera_stream and camera_stream.isOpened():
        camera_stream.release()
        camera_stream = None
    return jsonify({"message": "Camera stopped successfully"}), 200

# Run Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
