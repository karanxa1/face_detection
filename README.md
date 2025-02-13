Human Detection System - Flask & OpenCV
This is a web-based human detection system using Flask and OpenCV. It allows:
✅ Live webcam-based human detection
✅ Uploading videos for human detection processing
✅ Processing videos and playing the processed output
✅ Stopping the live webcam feed
✅ Video encoding using H.264 for compatibility

📌 Features
Webcam Mode: Live human detection using OpenCV, Button to start and stop the webcam
Upload Video Mode: Upload a video file, Process it to detect humans, Watch the processed video after completion
🚀 How to Run
1️⃣ Install Dependencies
Make sure you have Python 3.7+ installed. Then install required packages:

bash
Copy
Edit
pip install flask flask-cors opencv-python numpy
2️⃣ Run the Flask App
bash
Copy
Edit
python app.py
The app will start on: 📍 http://127.0.0.1:5000/

3️⃣ Using the App
Open your browser and go to http://127.0.0.1:5000/
Click Start Camera to begin human detection using your webcam
Click Stop Camera to stop the webcam feed
Upload a video file and click Process Video to detect humans in it
Watch the processed video in the output tab after processing is completed
📂 Project Structure
bash
Copy
Edit
/human-detection-app  
│── app.py               # Flask backend  
│── templates/  
│   ├── index.html       # Frontend HTML file  
│── static/  
│   ├── uploads/         # Folder for uploaded videos  
│   ├── processed/       # Folder for processed videos  
🛠 Requirements
Python 3.7+
Flask
OpenCV (cv2)
NumPy
📢 Notes
The system uses H.264 encoding for better browser compatibility
Make sure your browser allows camera access
Processed videos are saved in the /static/processed/ folder
📧 Contact
