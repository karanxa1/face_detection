from flask import Flask, render_template, request, Response
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load OpenCV's face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    if 'frame' not in request.files:
        return "No frame received", 400

    file = request.files['frame'].read()
    np_img = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert processed frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    return Response(frame_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
