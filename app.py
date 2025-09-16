from flask import Flask, Response, render_template_string, jsonify
import cv2
import mediapipe as mp
import math
import numpy as np
import time
import torch
from PIL import Image
from torchvision import transforms
import json
from collections import deque, defaultdict
import threading
import statistics

# Flask app
app = Flask(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load PyTorch model
pth_model = torch.jit.load('torchscript_model_0_66_37_wo_gl.pth')
pth_model.to(device)
pth_model.eval()

# Emotion label dictionary
DICT_EMO = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
EMOTION_COLORS = {
    'Neutral': '#64748B',
    'Happiness': '#22C55E',
    'Sadness': '#3B82F6',
    'Surprise': '#F59E0B',
    'Fear': '#8B5CF6',
    'Disgust': '#EF4444',
    'Anger': '#DC2626'
}

# Eye tracking landmarks (MediaPipe face mesh indices)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Global variables for statistics
emotion_history = deque(maxlen=50)
current_emotions = []
current_gaze_data = []
fps_history = deque(maxlen=30)
attentiveness_history = deque(maxlen=100)  # Store attentiveness scores
stats_lock = threading.Lock()

# Video stream
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://10.125.59.83:8080/video")

# MediaPipe face mesh with iris tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ========== Utility Functions ==========

def norm_coordinates(normalized_x, normalized_y, image_width, image_height):
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def get_box(fl, w, h):
    idx_to_coors = {}
    for idx, landmark in enumerate(fl.landmark):
        landmark_px = norm_coordinates(landmark.x, landmark.y, w, h)
        if landmark_px:
            idx_to_coors[idx] = landmark_px
    coords = np.asarray(list(idx_to_coors.values()))
    x_min, y_min = np.min(coords, axis=0)
    endX, endY = np.max(coords, axis=0)
    return max(0, x_min), max(0, y_min), min(w - 1, endX), min(h - 1, endY)

def calculate_eye_aspect_ratio(eye_landmarks):
    """Calculate eye aspect ratio to detect blinks"""
    if len(eye_landmarks) < 6:
        return 0.3

    try:
        eye_points = np.array(eye_landmarks[:6])
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        if C > 0:
            ear = (A + B) / (2.0 * C)
            return max(0.1, min(1.0, ear))
        return 0.3
    except Exception as e:
        return 0.3

def get_gaze_direction(eye_center, iris_center):
    """Calculate gaze direction based on iris position relative to eye center"""
    if eye_center is None or iris_center is None or len(eye_center) != 2 or len(iris_center) != 2:
        return "Center"
    
    # Calculate the displacement
    dx = iris_center[0] - eye_center[0]
    dy = iris_center[1] - eye_center[1]
    
    # Define thresholds for determining direction
    threshold_x = 2  # Horizontal threshold
    threshold_y = 2  # Vertical threshold
    
    # Determine primary direction
    if abs(dx) > threshold_x or abs(dy) > threshold_y:
        if abs(dx) > abs(dy):
            # Horizontal movement is dominant
            if dx > threshold_x:
                return "Right"
            elif dx < -threshold_x:
                return "Left"
        else:
            # Vertical movement is dominant
            if dy > threshold_y:
                return "Down"
            elif dy < -threshold_y:
                return "Up"
    
    return "Center"

def calculate_attentiveness_score(gaze_direction, is_blinking, ear_value, emotion):
    """Calculate attentiveness score based on gaze, blinking, and emotion"""
    score = 0.0
    
    # Gaze direction scoring (40% weight)
    gaze_scores = {
        "Center": 1.0,
        "Up": 0.6,
        "Down": 0.3,  # Looking down often indicates distraction
        "Left": 0.5,
        "Right": 0.5
    }
    score += gaze_scores.get(gaze_direction, 0.5) * 0.4
    
    # Blinking scoring (25% weight)
    if is_blinking:
        score += 0.1 * 0.25  # Blinking reduces attentiveness slightly
    else:
        # Normal eye openness indicates attention
        if ear_value > 0.25:
            score += 1.0 * 0.25
        else:
            score += 0.3 * 0.25  # Very low EAR might indicate fatigue
    
    # Emotion scoring (35% weight)
    emotion_scores = {
        'Happiness': 0.9,
        'Surprise': 0.8,
        'Neutral': 0.7,
        'Fear': 0.6,
        'Anger': 0.4,
        'Sadness': 0.3,
        'Disgust': 0.2
    }
    score += emotion_scores.get(emotion, 0.5) * 0.35
    
    return min(1.0, max(0.0, score))

def get_eye_tracking_data(landmarks, w, h):
    """Extract gaze direction and blink data without any visual drawing"""
    gaze_data = []
    if landmarks is None:
        return gaze_data

    try:
        landmark_points = []
        for landmark in landmarks.landmark:
            x, y = norm_coordinates(landmark.x, landmark.y, w, h)
            landmark_points.append((x, y))

        if len(landmark_points) < 478:
            return gaze_data

        # Initialize default values
        left_gaze = "Center"
        right_gaze = "Center"
        avg_ear = 0.3
        is_blinking = False

        # Left eye processing
        try:
            left_eye_points = [landmark_points[i] for i in LEFT_EYE if i < len(landmark_points)]
            if len(left_eye_points) >= 12:
                left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
                left_iris_points = [landmark_points[i] for i in LEFT_IRIS if i < len(landmark_points)]
                if len(left_iris_points) >= 4:
                    left_iris_center = np.mean(left_iris_points, axis=0).astype(int)
                    left_gaze = get_gaze_direction(left_eye_center, left_iris_center)
        except Exception:
            pass

        # Right eye processing
        try:
            right_eye_points = [landmark_points[i] for i in RIGHT_EYE if i < len(landmark_points)]
            if len(right_eye_points) >= 12:
                right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
                right_iris_points = [landmark_points[i] for i in RIGHT_IRIS if i < len(landmark_points)]
                if len(right_iris_points) >= 4:
                    right_iris_center = np.mean(right_iris_points, axis=0).astype(int)
                    right_gaze = get_gaze_direction(right_eye_center, right_iris_center)
        except Exception:
            pass

        # Blink detection
        try:
            left_eye_points_blink = [landmark_points[i] for i in LEFT_EYE[:6] if i < len(landmark_points)]
            right_eye_points_blink = [landmark_points[i] for i in RIGHT_EYE[:6] if i < len(landmark_points)]
            if len(left_eye_points_blink) >= 6 and len(right_eye_points_blink) >= 6:
                left_ear = calculate_eye_aspect_ratio(left_eye_points_blink)
                right_ear = calculate_eye_aspect_ratio(right_eye_points_blink)
                avg_ear = (left_ear + right_ear) / 2.0
                is_blinking = avg_ear < 0.23
        except Exception:
            pass

        # Combine gaze directions (use the most common direction or left eye as primary)
        combined_gaze = left_gaze
        if left_gaze != right_gaze and right_gaze != "Center":
            if left_gaze == "Center":
                combined_gaze = right_gaze
            # If both eyes look in different directions, prefer horizontal movements
            elif left_gaze in ["Left", "Right"] and right_gaze in ["Up", "Down"]:
                combined_gaze = left_gaze
            elif right_gaze in ["Left", "Right"] and left_gaze in ["Up", "Down"]:
                combined_gaze = right_gaze

        gaze_data.append({
            'left_gaze': left_gaze,
            'right_gaze': right_gaze,
            'combined_gaze': combined_gaze,
            'blinking': is_blinking,
            'ear': float(avg_ear),
            'timestamp': time.time()
        })

    except Exception as e:
        print(f"Eye tracking error: {e}")

    return gaze_data

def display_EMO_PRED(img, box, label='', confidence=0.0, gaze_direction="Center", is_blinking=False, attentiveness=0.0):
    """Display emotion with gaze direction and attentiveness above emotion without eye highlighting"""
    emotion_color_hex = EMOTION_COLORS.get(label, '#64748B')
    emotion_color_rgb = tuple(int(emotion_color_hex[i:i+2], 16) for i in (1, 3, 5))
    emotion_color_bgr = (emotion_color_rgb[2], emotion_color_rgb[1], emotion_color_rgb[0])

    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    # Modern rounded rectangle for face detection
    thickness = 3
    cv2.rectangle(img, (x1, y1), (x2, y2), emotion_color_bgr, thickness)

    # Confidence bar
    bar_width = x2 - x1
    bar_height = 6
    bar_y = y1 - 15

    # Background bar
    cv2.rectangle(img, (x1, bar_y), (x1 + bar_width, bar_y + bar_height), (40, 40, 40), -1)
    # Confidence fill
    cv2.rectangle(img, (x1, bar_y), (x1 + int(bar_width * confidence), bar_y + bar_height), emotion_color_bgr, -1)

    # Text styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thickness = 2

    # Create attentiveness status
    attentive_status = "Attentive" if attentiveness > 0.6 else "Distracted"
    attentive_color = (0, 255, 0) if attentiveness > 0.6 else (0, 100, 255)

    # Create gaze direction text
    gaze_text = f"Looking {gaze_direction.lower()}"
    if is_blinking:
        gaze_text = f"Blinking ({gaze_direction.lower()})"

    # Calculate text sizes and positions
    attentive_text = f"{attentive_status} ({attentiveness:.1%})"
    attentive_text_size = cv2.getTextSize(attentive_text, font, font_scale * 0.75, font_thickness)[0]
    
    gaze_text_size = cv2.getTextSize(gaze_text, font, font_scale * 0.7, font_thickness)[0]
    emotion_text = f"{label} {confidence:.1%}"
    emotion_text_size = cv2.getTextSize(emotion_text, font, font_scale, font_thickness)[0]

    # Position texts (attentiveness at top, then gaze, then emotion)
    attentive_x = x1 + (bar_width - attentive_text_size[0]) // 2
    attentive_y = bar_y - 70  # Top position for attentiveness

    gaze_x = x1 + (bar_width - gaze_text_size[0]) // 2
    gaze_y = bar_y - 45  # Middle position for gaze

    emotion_x = x1 + (bar_width - emotion_text_size[0]) // 2
    emotion_y = bar_y - 8  # Bottom position for emotion

    # Draw attentiveness text
    cv2.putText(img, attentive_text, (attentive_x, attentive_y), font, font_scale * 0.75, attentive_color, font_thickness)

    # Gaze text color based on state
    if is_blinking:
        gaze_color = (0, 215, 255)  # Orange for blinking
    elif gaze_direction == "Center":
        gaze_color = (255, 255, 255)  # White for center
    else:
        gaze_color = (0, 255, 255)   # Yellow for directional gaze

    # Draw gaze direction text
    cv2.putText(img, gaze_text, (gaze_x, gaze_y), font, font_scale * 0.7, gaze_color, font_thickness)

    # Draw emotion text
    cv2.putText(img, emotion_text, (emotion_x, emotion_y), font, font_scale, (255, 255, 255), font_thickness)

    return img

def display_FPS(img, fps):
    """Modern FPS display"""
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = img.shape[1] - text_size[0] - 15
    y = 35

    # Modern background
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 10, y - text_size[1] - 5), (x + text_size[0] + 10, y + 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # FPS color coding
    fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 15 else (0, 0, 255)
    cv2.putText(img, text, (x, y), font, font_scale, fps_color, thickness)

    return img

class PreprocessInput(torch.nn.Module):
    def init(self):
        super().init()
    def forward(self, x):
        x = x.to(torch.float32)
        x = torch.flip(x, dims=(0,))
        x[0, :, :] -= 91.4953
        x[1, :, :] -= 103.8827
        x[2, :, :] -= 131.0912
        return x

def pth_processing(img_pil):
    transform = transforms.Compose([
        transforms.PILToTensor(),
        PreprocessInput()
    ])
    img_pil = img_pil.resize((224, 224), Image.Resampling.NEAREST)
    img_tensor = transform(img_pil)
    return torch.unsqueeze(img_tensor, 0)

# ========== Flask Routes ==========

def gen_frames():
    global emotion_history, current_emotions, current_gaze_data, fps_history, attentiveness_history

    while cap.isOpened():
        t1 = time.time()
        success, frame = cap.read()
        if not success or frame is None:
            continue

        w, h = frame.shape[1], frame.shape[0]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        frame_emotions = []
        frame_gaze_data = []
        frame_attentiveness = []

        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                # Emotion detection
                startX, startY, endX, endY = get_box(fl, w, h)
                face_crop = rgb[startY:endY, startX:endX]
                if face_crop.size == 0:
                    continue

                face_tensor = pth_processing(Image.fromarray(face_crop)).to(device)
                output = torch.nn.functional.softmax(pth_model(face_tensor), dim=1).cpu().detach().numpy()
                emotion_idx = np.argmax(output)
                label = DICT_EMO[emotion_idx]
                confidence = output[0][emotion_idx]

                # Get gaze direction (no visual drawing, only data extraction)
                gaze_data = get_eye_tracking_data(fl, w, h)
                if gaze_data:
                    gaze_direction = gaze_data[0]['combined_gaze']
                    is_blinking = gaze_data[0]['blinking']
                    ear_value = gaze_data[0]['ear']
                    
                    # Calculate attentiveness score
                    attentiveness = calculate_attentiveness_score(gaze_direction, is_blinking, ear_value, label)
                    frame_attentiveness.append(attentiveness)
                    
                    frame_gaze_data.append({
                        'left_gaze': gaze_data[0]['left_gaze'],
                        'right_gaze': gaze_data[0]['right_gaze'],
                        'combined_gaze': gaze_direction,
                        'blinking': is_blinking,
                        'ear': ear_value,
                        'attentiveness': attentiveness,
                        'timestamp': time.time()
                    })
                else:
                    gaze_direction = "Center"
                    is_blinking = False
                    attentiveness = calculate_attentiveness_score(gaze_direction, is_blinking, 0.3, label)
                    frame_attentiveness.append(attentiveness)

                frame_emotions.append({
                    'emotion': label,
                    'confidence': float(confidence),
                    'gaze_direction': gaze_direction,
                    'is_blinking': is_blinking,
                    'attentiveness': attentiveness,
                    'timestamp': time.time()
                })

                # Display emotion with gaze direction and attentiveness above (NO eye highlighting on video)
                frame = display_EMO_PRED(frame, (startX, startY, endX, endY), label, confidence, gaze_direction, is_blinking, attentiveness)

        # Update statistics
        with stats_lock:
            current_emotions = frame_emotions
            current_gaze_data = frame_gaze_data
            for emotion_data in frame_emotions:
                emotion_history.append(emotion_data)
            
            # Add attentiveness scores to history
            if frame_attentiveness:
                avg_attentiveness = sum(frame_attentiveness) / len(frame_attentiveness)
                attentiveness_history.append(avg_attentiveness)

        fps = 1 / (time.time() - t1)
        fps_history.append(fps)
        frame = display_FPS(frame, fps)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    with stats_lock:
        # Current FPS
        current_fps = list(fps_history)[-1] if fps_history else 0
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

        # Recent emotion trend
        recent_emotions = list(emotion_history)[-10:]
        
        # Attentiveness statistics
        current_attentiveness = 0
        avg_attentiveness = 0
        if attentiveness_history:
            current_attentiveness = list(attentiveness_history)[-1]
            avg_attentiveness = sum(attentiveness_history) / len(attentiveness_history)
        
        # Overall attentiveness status
        attentiveness_status = "Attentive" if current_attentiveness > 0.6 else "Distracted"

        return jsonify({
            'current_emotions': current_emotions,
            'current_gaze_data': current_gaze_data,
            'current_fps': round(current_fps, 1),
            'average_fps': round(avg_fps, 1),
            'recent_emotions': recent_emotions,
            'emotion_colors': EMOTION_COLORS,
            'current_attentiveness': round(current_attentiveness, 3),
            'average_attentiveness': round(avg_attentiveness, 3),
            'attentiveness_status': attentiveness_status,
            'attentiveness_history': list(attentiveness_history)[-20:]  # Last 20 points for chart
        })

@app.route('/api/reset')
def reset_stats():
    global emotion_history, fps_history, attentiveness_history
    with stats_lock:
        emotion_history.clear()
        fps_history.clear()
        attentiveness_history.clear()
    return jsonify({'status': 'success'})

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Vision Studio - Emotion & Attentiveness Tracking</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: radial-gradient(ellipse at top, #1e293b 0%, #0f172a 100%);
            min-height: 100vh;
            color: white;
            overflow-x: hidden;
        }

        .container {
            max-width: 1800px;
            margin: 0 auto;
            padding: 24px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            animation: slideDown 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .header h1 {
            font-size: clamp(2.5rem, 5vw, 4rem);
            font-weight: 800;
            background: linear-gradient(135deg, #64FFDA 0%, #1DE9B6 25%, #00BCD4 50%, #2196F3 75%, #3F51B5 100%);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradientShift 3s ease infinite;
            margin-bottom: 12px;
        }

        .header p {
            font-size: 1.25rem;
            opacity: 0.85;
            font-weight: 300;
        }

        .main-layout {
            display: grid;
            grid-template-columns: 1fr 400px 300px;
            gap: 32px;
            margin-bottom: 32px;
        }

        .video-section {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 32px;
            box-shadow:
                0 25px 50px -12px rgba(0, 0, 0, 0.5),
                0 0 0 1px rgba(255, 255, 255, 0.05);
            animation: slideLeft 0.8s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
        }

        .stats-panel {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 32px;
            box-shadow:
                0 25px 50px -12px rgba(0, 0, 0, 0.5),
                0 0 0 1px rgba(255, 255, 255, 0.05);
            animation: slideRight 0.8s cubic-bezier(0.16, 1, 0.3, 1);
            overflow-y: auto;
            max-height: 80vh;
        }

        .attentiveness-panel {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 32px;
            box-shadow:
                0 25px 50px -12px rgba(0, 0, 0, 0.5),
                0 0 0 1px rgba(255, 255, 255, 0.05);
            animation: slideRight 0.8s cubic-bezier(0.16, 1, 0.3, 1);
            animation-delay: 0.2s;
            overflow-y: auto;
            max-height: 80vh;
        }

        .video-container {
            position: relative;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4);
            margin-bottom: 24px;
            background: #000;
        }

        .video-feed {
            width: 100%;
            height: auto;
            display: block;
            transition: transform 0.3s ease;
        }

        .video-overlay {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            padding: 12px 18px;
            border-radius: 50px;
            font-size: 0.9rem;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .live-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #ff3b30;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 1.5s ease-in-out infinite;
        }

        .controls {
            display: flex;
            gap: 16px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            padding: 14px 28px;
            border-radius: 50px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-size: 0.95rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            position: relative;
            overflow: hidden;
        }

        .btn:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }

        .btn:active {
            transform: translateY(0) scale(0.98);
        }

        .btn.reset {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 24px;
            color: #64FFDA;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 32px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
        }

        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 6px;
            background: linear-gradient(135deg, #64FFDA, #1DE9B6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .metric-label {
            font-size: 0.85rem;
            opacity: 0.7;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .attentiveness-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            margin-bottom: 24px;
            transition: all 0.3s ease;
        }

        .attentiveness-card:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
        }

        .attentiveness-score {
            font-size: 3rem;
            font-weight: 900;
            margin-bottom: 8px;
        }

        .attentiveness-score.attentive {
            background: linear-gradient(135deg, #22C55E, #16A34A);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .attentiveness-score.distracted {
            background: linear-gradient(135deg, #EF4444, #DC2626);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .attentiveness-status {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 12px;
        }

        .attentiveness-status.attentive {
            color: #22C55E;
        }

        .attentiveness-status.distracted {
            color: #EF4444;
        }

        .attentiveness-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin: 16px 0;
        }

        .attentiveness-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .attentiveness-fill.attentive {
            background: linear-gradient(135deg, #22C55E, #16A34A);
        }

        .attentiveness-fill.distracted {
            background: linear-gradient(135deg, #EF4444, #DC2626);
        }

        .emotion-display {
            margin-bottom: 32px;
        }

        .emotion-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px;
            margin: 12px 0;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            border-left: 4px solid;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: slideInRight 0.5s ease-out;
        }

        .emotion-item:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateX(4px);
        }

        .emotion-name {
            font-weight: 600;
            font-size: 1rem;
        }

        .confidence-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .confidence-bar {
            width: 80px;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .confidence-text {
            font-size: 0.85rem;
            font-weight: 600;
            min-width: 40px;
        }

        .gaze-display {
            margin-bottom: 32px;
        }

        .gaze-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            margin: 8px 0;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            font-size: 0.9rem;
        }

        .blink-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-left: 8px;
        }

        .blink-indicator.active {
            background: #ff3b30;
            animation: pulse 0.5s ease-in-out infinite;
        }

        .blink-indicator.inactive {
            background: #30d158;
        }

        .chart-container {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 20px;
            height: 250px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 24px;
        }

        .no-data {
            text-align: center;
            color: rgba(255, 255, 255, 0.5);
            padding: 40px 20px;
            font-size: 0.95rem;
        }

        .floating-particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }

        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideLeft {
            from { opacity: 0; transform: translateX(-50px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes slideRight {
            from { opacity: 0; transform: translateX(50px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(0.95); }
        }

        @media (max-width: 1400px) {
            .main-layout {
                grid-template-columns: 1fr 350px;
                gap: 24px;
            }
            
            .attentiveness-panel {
                grid-column: 2;
                grid-row: 1;
            }
        }

        @media (max-width: 1024px) {
            .main-layout {
                grid-template-columns: 1fr;
                gap: 24px;
            }

            .metrics-grid {
                grid-template-columns: 1fr;
            }

            .container {
                padding: 16px;
            }
        }

        @media (max-width: 640px) {
            .controls {
                flex-direction: column;
            }

            .btn {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="floating-particles"></div>

    <div class="container">
        <div class="header">
            <h1>AI Vision Studio</h1>
            <p>Advanced Emotion Recognition, Eye Tracking & Attentiveness Analysis</p>
        </div>

        <div class="main-layout">
            <div class="video-section">
                <div class="video-container">
                    <img src="/video_feed" class="video-feed" alt="Live Video Feed">
                    <div class="video-overlay">
                        <span class="live-dot"></span>
                        LIVE STREAM
                    </div>
                </div>
                <div class="controls">
                    <button class="btn" onclick="toggleFullscreen()">Fullscreen</button>
                    <button class="btn reset" onclick="resetStats()">Reset</button>
                </div>
            </div>

            <div class="stats-panel">
                <div class="section-title">
                    Live Metrics
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="current-fps">0</div>
                        <div class="metric-label">FPS</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="avg-fps">0</div>
                        <div class="metric-label">Avg FPS</div>
                    </div>
                </div>

                <div class="emotion-display">
                    <div class="section-title">
                        Current Emotions
                    </div>
                    <div id="current-emotions-list"></div>
                </div>

                <div class="gaze-display">
                    <div class="section-title">
                        Eye Tracking
                    </div>
                    <div id="gaze-data-list"></div>
                </div>

                <div class="chart-container">
                    <canvas id="emotionChart"></canvas>
                </div>
            </div>

            <div class="attentiveness-panel">
                <div class="section-title">
                    Attentiveness Monitor
                </div>

                <div class="attentiveness-card">
                    <div class="attentiveness-score" id="current-attentiveness-score">0%</div>
                    <div class="attentiveness-status" id="attentiveness-status">Analyzing...</div>
                    <div class="attentiveness-bar">
                        <div class="attentiveness-fill" id="attentiveness-fill" style="width: 0%;"></div>
                    </div>
                    <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 8px;">
                        Real-time attention level
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-value" id="avg-attentiveness">0%</div>
                    <div class="metric-label">Session Average</div>
                </div>

                <div class="section-title" style="margin-top: 32px;">
                    Attention Trend
                </div>

                <div class="chart-container">
                    <canvas id="attentivenessChart"></canvas>
                </div>

                <div class="section-title" style="margin-top: 32px;">
                    Analysis Factors
                </div>

                <div style="background: rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 16px; font-size: 0.85rem; line-height: 1.6;">
                    <div style="margin-bottom: 12px;"><strong>Gaze Direction (40%):</strong> Center gaze indicates highest attention</div>
                    <div style="margin-bottom: 12px;"><strong>Blinking Pattern (25%):</strong> Normal blinking vs excessive blinking</div>
                    <div><strong>Emotional State (35%):</strong> Positive emotions suggest better engagement</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let emotionChart;
        let attentivenessChart;
        let emotionHistory = [];
        const maxHistoryPoints = 20;

        // Initialize charts
        function initCharts() {
            // Emotion Chart
            const emotionCtx = document.getElementById('emotionChart').getContext('2d');
            emotionChart = new Chart(emotionCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 300,
                        easing: 'easeInOutCubic'
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: 'rgba(255, 255, 255, 0.8)',
                                font: { size: 11 },
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: 'white',
                            bodyColor: 'white',
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                            borderWidth: 1,
                            cornerRadius: 8
                        }
                    },
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            min: 0,
                            max: 1,
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.6)',
                                font: { size: 10 }
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)',
                                drawBorder: false
                            }
                        }
                    }
                }
            });

            // Attentiveness Chart
            const attentionCtx = document.getElementById('attentivenessChart').getContext('2d');
            attentivenessChart = new Chart(attentionCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Attentiveness',
                        data: [],
                        borderColor: '#22C55E',
                        backgroundColor: '#22C55E20',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#22C55E',
                        pointBorderColor: 'white',
                        pointBorderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 300,
                        easing: 'easeInOutCubic'
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: 'white',
                            bodyColor: 'white',
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                            borderWidth: 1,
                            cornerRadius: 8,
                            callbacks: {
                                label: function(context) {
                                    return 'Attentiveness: ' + (context.parsed.y * 100).toFixed(1) + '%';
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            min: 0,
                            max: 1,
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.6)',
                                font: { size: 10 },
                                callback: function(value) {
                                    return (value * 100).toFixed(0) + '%';
                                }
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)',
                                drawBorder: false
                            }
                        }
                    }
                }
            });
        }

        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    // Update FPS metrics
                    document.getElementById('current-fps').textContent = data.current_fps || 0;
                    document.getElementById('avg-fps').textContent = data.average_fps || 0;

                    // Update current emotions
                    updateCurrentEmotions(data.current_emotions, data.emotion_colors);

                    // Update gaze data
                    updateGazeData(data.current_gaze_data);

                    // Update emotion history chart
                    updateEmotionChart(data.recent_emotions, data.emotion_colors);

                    // Update attentiveness
                    updateAttentiveness(data);
                })
                .catch(error => console.error('Error fetching stats:', error));
        }

        function updateCurrentEmotions(emotions, colors) {
            const container = document.getElementById('current-emotions-list');

            if (!emotions || emotions.length === 0) {
                container.innerHTML = '<div class="no-data">No faces detected</div>';
                return;
            }

            container.innerHTML = '';
            emotions.forEach((emotion, index) => {
                const emotionDiv = document.createElement('div');
                emotionDiv.className = 'emotion-item';
                emotionDiv.style.borderLeftColor = colors[emotion.emotion] || '#64748B';
                emotionDiv.style.animationDelay = ${index * 0.1}s;

                emotionDiv.innerHTML = `
                    <span class="emotion-name">${emotion.emotion}</span>
                    <div class="confidence-container">
                        <div class="confidence-bar">
                            <div class="confidence-fill"
                                 style="width: ${emotion.confidence * 100}%; background-color: ${colors[emotion.emotion] || '#64748B'};"></div>
                        </div>
                        <span class="confidence-text">${(emotion.confidence * 100).toFixed(0)}%</span>
                    </div>
                `;
                container.appendChild(emotionDiv);
            });
        }

        function updateGazeData(gazeData) {
            const container = document.getElementById('gaze-data-list');

            if (!gazeData || gazeData.length === 0) {
                container.innerHTML = '<div class="no-data">No eye data available</div>';
                return;
            }

            container.innerHTML = '';
            gazeData.forEach((gaze, index) => {
                const gazeDiv = document.createElement('div');
                gazeDiv.className = 'gaze-item';
                gazeDiv.style.animationDelay = ${index * 0.05}s;

                // Format gaze direction display
                const combinedGaze = gaze.combined_gaze || 'Center';
                const gazeText = combinedGaze.charAt(0).toUpperCase() + combinedGaze.slice(1).toLowerCase();
                
                gazeDiv.innerHTML = `
                    <div>
                        <strong>Looking:</strong> ${gazeText}
                        <br>
                        <small style="opacity: 0.7;">L: ${gaze.left_gaze || 'Center'} | R: ${gaze.right_gaze || 'Center'}</small>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.8rem;">Blink Status</span>
                        <span class="blink-indicator ${gaze.blinking ? 'active' : 'inactive'}"></span>
                        <br>
                        <span style="font-size: 0.75rem; opacity: 0.7;">EAR: ${(gaze.ear || 0).toFixed(2)}</span>
                    </div>
                `;
                container.appendChild(gazeDiv);
            });
        }

        function updateAttentiveness(data) {
            const currentScore = data.current_attentiveness || 0;
            const avgScore = data.average_attentiveness || 0;
            const status = data.attentiveness_status || 'Analyzing...';

            // Update current attentiveness display
            const scoreElement = document.getElementById('current-attentiveness-score');
            const statusElement = document.getElementById('attentiveness-status');
            const fillElement = document.getElementById('attentiveness-fill');

            scoreElement.textContent = ${(currentScore * 100).toFixed(0)}%;
            statusElement.textContent = status;

            // Update colors based on attentiveness level
            const isAttentive = currentScore > 0.6;
            scoreElement.className = attentiveness-score ${isAttentive ? 'attentive' : 'distracted'};
            statusElement.className = attentiveness-status ${isAttentive ? 'attentive' : 'distracted'};
            fillElement.className = attentiveness-fill ${isAttentive ? 'attentive' : 'distracted'};
            fillElement.style.width = ${currentScore * 100}%;

            // Update average
            document.getElementById('avg-attentiveness').textContent = ${(avgScore * 100).toFixed(0)}%;

            // Update attentiveness chart
            if (data.attentiveness_history && data.attentiveness_history.length > 0) {
                const history = data.attentiveness_history;
                attentivenessChart.data.labels = Array.from({length: history.length}, (_, i) => i + 1);
                attentivenessChart.data.datasets[0].data = history;
                
                // Update line color based on recent trend
                const recentAvg = history.slice(-5).reduce((a, b) => a + b, 0) / Math.min(5, history.length);
                const lineColor = recentAvg > 0.6 ? '#22C55E' : '#EF4444';
                attentivenessChart.data.datasets[0].borderColor = lineColor;
                attentivenessChart.data.datasets[0].pointBackgroundColor = lineColor;
                
                attentivenessChart.update('none');
            }
        }

        function updateEmotionChart(recentEmotions, colors) {
            if (!recentEmotions || recentEmotions.length === 0) return;

            // Group emotions by type
            const emotionGroups = {};
            recentEmotions.forEach((emotion, index) => {
                if (!emotionGroups[emotion.emotion]) {
                    emotionGroups[emotion.emotion] = [];
                }
                emotionGroups[emotion.emotion].push({
                    x: index,
                    y: emotion.confidence
                });
            });

            // Create datasets
            const datasets = Object.keys(emotionGroups).map(emotionName => ({
                label: emotionName,
                data: emotionGroups[emotionName],
                borderColor: colors[emotionName] || '#64748B',
                backgroundColor: (colors[emotionName] || '#64748B') + '20',
                borderWidth: 2,
                fill: false,
                tension: 0.4,
                pointRadius: 3,
                pointHoverRadius: 5,
                pointBackgroundColor: colors[emotionName] || '#64748B',
                pointBorderColor: 'white',
                pointBorderWidth: 1
            }));

            // Update chart
            emotionChart.data.labels = Array.from({length: recentEmotions.length}, (_, i) => i + 1);
            emotionChart.data.datasets = datasets;
            emotionChart.update('none');
        }

        function resetStats() {
            fetch('/api/reset', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        // Clear charts
                        emotionChart.data.labels = [];
                        emotionChart.data.datasets = [];
                        emotionChart.update();

                        attentivenessChart.data.labels = [];
                        attentivenessChart.data.datasets[0].data = [];
                        attentivenessChart.update();

                        // Clear displays
                        document.getElementById('current-emotions-list').innerHTML = '<div class="no-data">No faces detected</div>';
                        document.getElementById('gaze-data-list').innerHTML = '<div class="no-data">No eye data available</div>';
                        
                        // Reset attentiveness display
                        document.getElementById('current-attentiveness-score').textContent = '0%';
                        document.getElementById('attentiveness-status').textContent = 'Analyzing...';
                        document.getElementById('attentiveness-fill').style.width = '0%';
                        document.getElementById('avg-attentiveness').textContent = '0%';

                        // Success feedback
                        const btn = event.target;
                        const originalText = btn.textContent;
                        const originalClass = btn.className;

                        btn.textContent = 'Reset Complete';
                        btn.style.background = 'linear-gradient(135deg, #30d158, #28a745)';

                        setTimeout(() => {
                            btn.textContent = originalText;
                            btn.className = originalClass;
                            btn.style.background = '';
                        }, 2000);
                    }
                })
                .catch(error => console.error('Error resetting stats:', error));
        }

        function toggleFullscreen() {
            const videoContainer = document.querySelector('.video-container');

            if (!document.fullscreenElement) {
                videoContainer.requestFullscreen().then(() => {
                    videoContainer.style.position = 'fixed';
                    videoContainer.style.top = '0';
                    videoContainer.style.left = '0';
                    videoContainer.style.width = '100vw';
                    videoContainer.style.height = '100vh';
                    videoContainer.style.zIndex = '9999';
                    videoContainer.style.borderRadius = '0';
                }).catch(err => console.error('Fullscreen error:', err));
            } else {
                document.exitFullscreen();
            }
        }

        // Handle fullscreen changes
        document.addEventListener('fullscreenchange', () => {
            const videoContainer = document.querySelector('.video-container');
            if (!document.fullscreenElement) {
                videoContainer.style.position = 'relative';
                videoContainer.style.top = 'auto';
                videoContainer.style.left = 'auto';
                videoContainer.style.width = 'auto';
                videoContainer.style.height = 'auto';
                videoContainer.style.zIndex = 'auto';
                videoContainer.style.borderRadius = '20px';
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (event) => {
            if (event.ctrlKey || event.metaKey) {
                switch(event.key.toLowerCase()) {
                    case 'f':
                        event.preventDefault();
                        toggleFullscreen();
                        break;
                    case 'r':
                        event.preventDefault();
                        resetStats();
                        break;
                }
            }
        });

        // Connection status monitoring
        function checkConnection() {
            fetch('/api/stats')
                .then(() => {
                    document.querySelector('.live-dot').style.background = '#ff3b30';
                    document.querySelector('.live-dot').style.animation = 'pulse 1.5s ease-in-out infinite';
                })
                .catch(() => {
                    document.querySelector('.live-dot').style.background = '#ffa502';
                    document.querySelector('.live-dot').style.animation = 'none';
                });
        }

        // Video error handling
        function handleVideoError() {
            const videoFeed = document.querySelector('.video-feed');
            videoFeed.addEventListener('error', () => {
                console.warn('Video feed error - attempting reconnection...');
                setTimeout(() => {
                    videoFeed.src = '/video_feed?' + Date.now();
                }, 3000);
            });
        }

        // Floating particles animation
        function createFloatingParticles() {
            const container = document.querySelector('.floating-particles');
            const particleCount = 15;

            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.style.cssText = `
                    position: absolute;
                    width: ${Math.random() * 4 + 1}px;
                    height: ${Math.random() * 4 + 1}px;
                    background: rgba(100, 255, 218, ${Math.random() * 0.3 + 0.1});
                    border-radius: 50%;
                    left: ${Math.random() * 100}%;
                    top: ${Math.random() * 100}%;
                    animation: float ${Math.random() * 20 + 10}s linear infinite;
                `;
                container.appendChild(particle);
            }

            // Add floating animation keyframes
            const style = document.createElement('style');
            style.textContent = `
                @keyframes float {
                    0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
                    10% { opacity: 1;90% { opacity: 1; }
                    100% { transform: translateY(-20vh) rotate(360deg); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }

        // Performance monitoring
        function monitorPerformance() {
            const startTime = performance.now();
            let frameCount = 0;
            let lastTime = startTime;

            function checkPerformance() {
                frameCount++;
                const currentTime = performance.now();
                
                if (currentTime - lastTime >= 5000) { // Check every 5 seconds
                    const fps = (frameCount * 1000) / (currentTime - lastTime);
                    if (fps < 10) {
                        console.warn('Low performance detected. Consider reducing update frequency.');
                    }
                    frameCount = 0;
                    lastTime = currentTime;
                }
                
                requestAnimationFrame(checkPerformance);
            }
            
            requestAnimationFrame(checkPerformance);
        }

        // Initialize everything when DOM is loaded
        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            createFloatingParticles();
            handleVideoError();
            monitorPerformance();
            
            // Start periodic updates
            updateStats();
            setInterval(updateStats, 1000); // Update every second
            
            // Connection monitoring
            setInterval(checkConnection, 5000); // Check every 5 seconds
            
            // Initial connection check
            setTimeout(checkConnection, 1000);
            
            console.log('AI Vision Studio initialized successfully');
        });

        // Handle page visibility changes to optimize performance
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Page is hidden, reduce update frequency
                console.log('Page hidden - reducing update frequency');
            } else {
                // Page is visible, resume normal updates
                console.log('Page visible - resuming normal updates');
                updateStats(); // Immediate update when page becomes visible
            }
        });

        // Error handling for the entire application
        window.addEventListener('error', (event) => {
            console.error('Application error:', event.error);
        });

        // Unhandled promise rejection handling
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            event.preventDefault();
        });
    </script>
</body>
</html>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
