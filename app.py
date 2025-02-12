from flask import Flask, render_template, request, Response, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/'
PROCESSED_FOLDER = 'processed_videos/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load pre-trained human detection model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handles video upload and starts processing"""
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['video']
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    output_filename = os.path.join(PROCESSED_FOLDER, "processed_" + file.filename)
    process_video(filename, output_filename)

    return jsonify({'processed_video': output_filename})

def process_video(input_path, output_path):
    """Human detection on uploaded videos with proper encoding"""
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 encoding
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect humans
        boxes, _ = hog.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)

        # Draw rectangles around detected humans
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

@app.route('/video_feed')
def video_feed():
    """Live webcam video feed"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Human detection in live webcam mode"""
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect humans
        boxes, _ = hog.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)

        # Draw rectangles around detected humans
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/processed_videos/<filename>')
def processed_videos(filename):
    """Serve processed videos"""
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
