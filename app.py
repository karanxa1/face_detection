from flask import Flask, render_template, request, Response, jsonify, send_from_directory, url_for
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Directories
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Human Detection Model (HOG + SVM)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Global variable for webcam
camera_stream = None

@app.route('/')
def index():
    return render_template('index.html')

# 🔹 Webcam Feed
def generate_frames():
    global camera_stream
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

# 🔹 Upload & Process Video
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files['video']
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    processed_filename = os.path.join(PROCESSED_FOLDER, "processed_" + file.filename)
    process_video(filename, processed_filename)

    return jsonify({"processed_video": url_for('static', filename=f'processed/processed_{file.filename}')})

# 🔹 Process Video (H.264 Encoding)
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 Encoding
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.1)

        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
