﻿# CrashScope_deepLearning
CrashScope is an AI-powered car damage detection and classification system. Using deep learning and computer vision, it accurately identifies different types of damages from vehicle images — such as dents, broken headlights, damaged windshields, and more — making it useful for insurance assessment, vehicle inspection, and fleet management.

📌 Features
🔍 Detects multiple types of car damages from a single image

🧠 Built using YOLOv8 for real-time object detection

📊 Includes performance evaluation metrics (Precision, Recall, mAP, F1 Score, Accuracy)

🧪 Supports batch testing and result logging

📦 Containerized using Docker for easy deployment

📲 Telegram alert integration (optional for monitoring)

🏗️ Project Structure
bash
Copy
Edit
CrashScope/
│
├── data/                   # Dataset (images and annotations)
├── runs/                   # YOLOv8 training outputs
├── model/
│   ├── yolov8_custom.pt    # Trained YOLOv8 model
│   ├── monitor.py          # Prediction & evaluation script
│   └── config.yaml         # Configuration file for Telegram alerts
│
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Python dependencies
└── README.md               # You're here!
🧪 Model Details
Architecture: YOLOv8

Classes:

damaged door

damaged window

damaged headlight

damaged mirror

dent

damaged hood

damaged bumper

damaged windshield

Framework: PyTorch, Ultralytics YOLO

Accuracy: ~89% on validation set



