from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import torch
from torchvision import models, transforms
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# ====== Preprocessing & Image Quality Checks ======

FIXED_SIZE = (256, 256)

def resize_image(img):
    return cv2.resize(img, FIXED_SIZE)

def check_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Lower variance = more noisy / flat
    if laplacian_var < 50:
        return "High noise detected"
    return "Noise levels acceptable"

def check_white_pixels(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_ratio = np.sum(gray > 240) / gray.size
    if white_ratio > 0.4:
        return "Too many white/bright pixels (overexposed)"
    return "White pixel ratio acceptable"

def check_lighting(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    if mean_intensity < 50:
        return "Image too dark"
    elif mean_intensity > 200:
        return "Image too bright"
    return "Lighting conditions acceptable"

def check_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < 100:   # threshold tuneable
        return "Image is blurry"
    return "Sharpness acceptable"

def preprocess_and_validate(img):
    results = []

    # Step 1: Resize
    img_resized = resize_image(img)
    results.append("Resized to fixed size")

    # Step 2: Noise check
    results.append(check_noise(img_resized))

    # Step 3: White pixel check
    results.append(check_white_pixels(img_resized))

    # Step 4: Lighting check
    results.append(check_lighting(img_resized))

    # Step 5: Blur check
    results.append(check_blur(img_resized))

    return img_resized, results
#=====================================================

# ====== Health Check ======
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

# ====== Load Emotion Model =====
MODEL_PATH = "facial_emotion_detection_model.keras"
print(f"[INFO] Loading emotion model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Emotion model file not found at: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("[INFO] Emotion model loaded successfully")

class_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
print(f"[INFO] Emotion Class labels: {class_labels}")

# ====== Load DeepLab Human Segmentation Model =====
print("[INFO] Loading DeepLabV3 model for human detection...")
deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
print("[INFO] DeepLabV3 model loaded successfully")

# ====== Utility for Emotion Prediction =====
def predict_emotion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=(0, -1))
    preds = model.predict(input_tensor)
    class_idx = np.argmax(preds[0])
    return class_labels[class_idx]

# ====== Utilities for Human Detection =====
def preprocess_for_deeplab(img):
    trf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return trf(img).unsqueeze(0)

def get_person_mask(output):
    output_predictions = output['out'].squeeze(0).argmax(0).byte().cpu().numpy()
    person_mask = np.where(output_predictions == 15, 255, 0).astype(np.uint8)
    return person_mask

def get_alignment_feedback(bbox, img_shape):
    x, y, w, h = bbox
    img_h, img_w = img_shape[:2]
    center_x = x + w // 2

    feedback = []
    if w > img_w * 0.6:
        feedback.append("Too close to camera")
    elif w < img_w * 0.2:
        feedback.append("Too far from camera")
    
    if center_x < img_w * 0.3:
        feedback.append("Move right")
    elif center_x > img_w * 0.7:
        feedback.append("Move left")

    if not feedback:
        feedback.append("Aligned properly")

    return feedback

def detect_person_and_feedback(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess_for_deeplab(img_rgb)

    with torch.no_grad():
        output = deeplab(input_tensor)

    mask = get_person_mask(output)
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        feedback = get_alignment_feedback((x, y, w, h), img.shape)
        return True, feedback
    else:
        return False, ["No person detected"]

# ====== Flask Routes ======

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']
#     img_bytes = file.read()
#     img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

#     if img is None:
#         return jsonify({'error': 'Invalid image'}), 400

#     mood = predict_emotion(img)
#     return jsonify({'mood': mood})

# @app.route('/check_person', methods=['POST'])
# def check_person():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']
#     img_bytes = file.read()
#     img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

#     if img is None:
#         return jsonify({'error': 'Invalid image'}), 400

#     person_present, feedback = detect_person_and_feedback(img)

#     return jsonify({
#         'person_detected': person_present,
#         'feedback': feedback
#     })
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print("❌ [ERROR] Invalid image received for /predict")
        return jsonify({'error': 'Invalid image'}), 400

    # Run pre-checks
    img_resized, precheck_feedback = preprocess_and_validate(img)

    # Log pre-checks in terminal
    print("\n===== /predict Request =====")
    print(f"📷 Image size after resize: {img_resized.shape}")
    print("🛠 Pre-check results:")
    for msg in precheck_feedback:
        print(f"   - {msg}")

    # Predict emotion
    mood = predict_emotion(img_resized)
    print(f"😀 Predicted Mood: {mood}")
    print("============================\n")

    return jsonify({
        'mood': mood,
        'precheck_feedback': precheck_feedback
    })


@app.route('/check_person', methods=['POST'])
def check_person():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print("❌ [ERROR] Invalid image received for /check_person")
        return jsonify({'error': 'Invalid image'}), 400

    # Run pre-checks
    img_resized, precheck_feedback = preprocess_and_validate(img)

    # Human detection
    person_present, feedback = detect_person_and_feedback(img_resized)

    # Log in terminal
    print("\n===== /check_person Request =====")
    print(f"📷 Image size after resize: {img_resized.shape}")
    print("🛠 Pre-check results:")
    for msg in precheck_feedback:
        print(f"   - {msg}")

    if person_present:
        print("✅ Person detected")
    else:
        print("⚠️ No person detected")

    print("📌 Alignment feedback:")
    for msg in feedback:
        print(f"   - {msg}")
    print("===============================\n")

    return jsonify({
        'person_detected': person_present,
        'feedback': feedback,
        'precheck_feedback': precheck_feedback
    })


if __name__ == '__main__':
    print("[INFO] Starting Flask server on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
