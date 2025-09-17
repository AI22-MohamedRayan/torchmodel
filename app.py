import cv2
import time
import json
import numpy as np
from flask import Flask, render_template, Response
from deepface.basemodels import Facenet
from deepface.commons import functions
import torch
from ultralytics import YOLO

# ---- CONFIG ----
DB_PATH = "students"
TARGET_SIZE = (160, 160)
ALLOWED_EXT = (".jpg", ".jpeg", ".png")
THRESHOLD = 10
DISPLAY_TIME = 10

# ---- LOAD MODELS ----
print("🔄 Loading models...")
face_model = Facenet.loadModel()
print("✅ Facenet loaded!")

# Load YOLO (for face/person detection)
yolo_model = YOLO("yolov8n.pt")  # lightweight model
print("✅ YOLO model loaded!")

# Load attentiveness CNN model
cnn_model = torch.jit.load("torchscript_model_0_66_37_wo_gl.pth")
cnn_model.eval()
print("✅ Attentiveness CNN loaded!")

# ---- PREPARE DATABASE EMBEDDINGS ----
database_embeddings = []
for student_folder in functions.list_dir(DB_PATH):
    usn, name = student_folder.split("_", 1) if "_" in student_folder else ("UNKNOWN", student_folder)
    folder_path = f"{DB_PATH}/{student_folder}"
    embeddings = []
    for img_name in functions.list_dir(folder_path):
        if not img_name.lower().endswith(ALLOWED_EXT):
            continue
        img_path = f"{folder_path}/{img_name}"
        img = functions.preprocess_face(img_path, target_size=TARGET_SIZE, enforce_detection=False)
        if img is None:
            continue
        embedding = face_model.predict(img)
        embeddings.append(embedding)
    if embeddings:
        mean_emb = np.mean(embeddings, axis=0)
        database_embeddings.append((usn, name, mean_emb))

attendance = {}
display_tracker = {}

# ---- FACE RECOGNITION ----
def recognize_faces(frame, detections):
    identities = []
    for box in detections:
        x1, y1, x2, y2 = map(int, box[:4])
        w, h = x2 - x1, y2 - y1
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue

        img = functions.preprocess_face(face_img, target_size=TARGET_SIZE, enforce_detection=False)
        if img is None:
            continue

        frame_emb = face_model.predict(img)
        min_dist = float("inf")
        identity = "Unknown"

        for usn, name, db_emb in database_embeddings:
            dist = np.linalg.norm(frame_emb - db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = f"{usn} - {name}" if min_dist < THRESHOLD else "Unknown"

        identities.append(((x1, y1, w, h), identity, face_img))
    return identities

# ---- ATTENTIVENESS CLASSIFICATION ----
def classify_attentiveness(face_img):
    try:
        img = cv2.resize(face_img, (48, 48))  # adjust size to your CNN training
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = cnn_model(img)
            cls = torch.argmax(pred, dim=1).item()
        return cls  # map this to labels below
    except Exception:
        return -1

label_map = {0: "Attentive", 1: "Inattentive", 2: "Drowsy"}  # adjust based on training

# ---- VIDEO STREAM ----
def generate_frames():
    cap = cv2.VideoCapture(0)  # replace with NVR RTSP later
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()

        # YOLO detection (detect persons/faces)
        results = yolo_model(frame, verbose=False)
        detections = []
        for box in results[0].boxes.xyxy.cpu().numpy():
            detections.append(box)

        # Recognize + classify attentiveness
        recognized = recognize_faces(frame, detections)

        for (x, y, w, h), student_info, face_img in recognized:
            att_label = "Unknown"
            if student_info != "Unknown":
                usn, name = student_info.split(" - ")
                if usn not in attendance:
                    attendance[usn] = {"name": name, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                if usn not in display_tracker:
                    display_tracker[usn] = current_time

                if current_time - display_tracker[usn] < DISPLAY_TIME:
                    # classify attentiveness
                    cls = classify_attentiveness(face_img)
                    att_label = label_map.get(cls, "Unknown")

                    # draw box + info
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{student_info} | {att_label}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()

# ---- FLASK APP ----
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/save_attendance")
def save_attendance():
    with open("attendance.json", "w") as f:
        json.dump(attendance, f, indent=4)
    return {"status": "saved", "count": len(attendance)}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
