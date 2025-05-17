import argparse
import io
import os
import cv2
import base64
from flask import Flask, render_template, request, redirect, url_for, Response, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
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

# Video Streaming Generator Function
def generate():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference
        results = model(frame)
        annotated_frame = results[0].plot()

        # Encode the frame as JPEG
        _, jpeg = cv2.imencode('.jpg', annotated_frame)

        # Yield the frame in a format suitable for live streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

# Routes
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

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('image'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('image'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        img = Image.open(file_path)
        result = model.predict(source=img)[0]
        res_img = Image.fromarray(result.plot())

        image_byte_stream = io.BytesIO()
        res_img.save(image_byte_stream, format='PNG')
        image_byte_stream.seek(0)
        image_base64 = base64.b64encode(image_byte_stream.read()).decode('utf-8')

        processed_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.png')
        res_img.save(processed_img_path)

        return render_template('image.html', detection_results=image_base64, download_link='processed_image.png')

    except Exception as e:
        print(f"Error processing image: {e}")
        return redirect(url_for('image'))

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route("/predict_video", methods=["POST"])
def predict_video():
    if 'file' not in request.files:
        return redirect(url_for('video'))

    file = request.files['file']
    filename = secure_filename(file.filename)
    
    upload_dir = os.path.join('static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    if filename.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_filename = 'output.mp4'
        output_path = os.path.join(upload_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 25.0, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            res_plotted = results[0].plot()
            out.write(res_plotted)

        cap.release()
        out.release()

        # Send relative path to HTML
        video_path = f'uploads/{output_filename}'  # relative to 'static/'
        return render_template('video.html', video_path=video_path, video_download=output_filename)

    return redirect(url_for('video'))


@app.route('/stop', methods=['POST'])
def stop():
    return redirect(url_for('first'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLO models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
