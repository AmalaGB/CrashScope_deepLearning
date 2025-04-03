import argparse
import io
import os
import cv2
import base64
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image

# ✅ Load YOLO model (Fix: Removed weights_only=True)
model = YOLO('best.pt')

# Allowed image formats
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER = 'uploads'

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 🎥 Video Streaming Generator Function
def generate():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference
        results = model(frame)  # ✅ Fixed inference function
        annotated_frame = results[0].plot()

        # Encode the frame as JPEG
        _, jpeg = cv2.imencode('.jpg', annotated_frame)

        # Yield the frame in a format suitable for live streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

# 🏠 Home Route
# Home Route
@app.route('/')
@app.route('/first')
def first():
    return render_template("first.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/chart')
def chart():
    return render_template("chart.html")

@app.route('/performance')
def performance():
    return render_template("performance.html")

@app.route('/image')
def image():
    return render_template("image.html")

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Live video streaming route
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
# 📷 Image Upload & Processing
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('home'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        img = Image.open(file_path)
        results = model.predict(source=img)  # ✅ Fixed inference call
        res_img = Image.fromarray(results[0].plot())

        image_byte_stream = io.BytesIO()
        res_img.save(image_byte_stream, format='PNG')
        image_byte_stream.seek(0)
        image_base64 = base64.b64encode(image_byte_stream.read()).decode('utf-8')

        return render_template('image.html', detection_results=image_base64)

    except Exception as e:
        print(f"Error processing image: {e}")
        return redirect(url_for('home'))

# 🎥 Video Processing Route
@app.route("/predict_img", methods=["POST"])
def predict_img():
    if 'file' not in request.files:
        return redirect(url_for('video'))

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    if filename.endswith('.mp4'):
        video_path = file_path
        cap = cv2.VideoCapture(video_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 25.0, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            res_plotted = results[0].plot()
            out.write(res_plotted)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return redirect(url_for('video'))

    return redirect(url_for('video'))

# 🛑 Stop Video Streaming Route
@app.route('/stop', methods=['POST'])
def stop():
    return redirect(url_for('image'))

# Vercel requires a handler function
def handler(event, context):
    return app(event, context)

# 🚀 Start Flask App
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLO models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    
    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=args.port, debug=True)
