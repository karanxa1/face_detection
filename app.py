from flask import Flask, render_template, request, Response, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Upload folder setup
UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize HOG descriptor for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Global variable for processed video path
processed_video_path = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handles video upload and saves it."""
    global processed_video_path

    if 'video' not in request.files:
        return jsonify({'error': 'No video file received'}), 400

    file = request.files['video']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')
    file.save(filename)

    processed_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_video.mp4')

    # Start video processing
    process_video(filename, processed_video_path)

    return jsonify({'message': 'Video uploaded successfully', 'processed_video': '/processed_video'})

@app.route('/processed_video')
def get_processed_video():
    """Serves the processed video."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'processed_video.mp4')

def process_video(input_path, output_path):
    """Processes video for human detection and saves it with H.264 encoding."""
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 Encoding
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect humans in the frame
        boxes, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Draw bounding boxes for detected humans
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
