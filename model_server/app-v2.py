from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import torch
from torchvision import models, transforms
from tensorflow.keras.models import load_model
import time # Import time for logging

app = Flask(__name__)
CORS(app)

# ====== Preprocessing & Image Quality Checks ======

FIXED_SIZE = (256, 256)

def get_timestamp():
    """Helper function to get a formatted timestamp for logs."""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def resize_image(img):
    print(f"    [DEBUG {get_timestamp()}] Resizing image from {img.shape[:2]} to {FIXED_SIZE}")
    return cv2.resize(img, FIXED_SIZE)

def check_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"    [DEBUG {get_timestamp()}] Noise check (Laplacian Var): {laplacian_var:.2f}")
    if laplacian_var < 50:
        return "High noise detected"
    return "Noise levels acceptable"

def check_white_pixels(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_ratio = np.sum(gray > 240) / gray.size
    print(f"    [DEBUG {get_timestamp()}] White pixel ratio check: {white_ratio:.2f}")
    if white_ratio > 0.4:
        return "Too many white/bright pixels (overexposed)"
    return "White pixel ratio acceptable"

def check_lighting(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    print(f"    [DEBUG {get_timestamp()}] Lighting check (Mean Intensity): {mean_intensity:.2f}")
    if mean_intensity < 50:
        return "Image too dark"
    elif mean_intensity > 200:
        return "Image too bright"
    return "Lighting conditions acceptable"

def check_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"    [DEBUG {get_timestamp()}] Blur check (Laplacian Var): {fm:.2f}")
    if fm < 100:  # threshold tuneable
        return "Image is blurry"
    return "Sharpness acceptable"

# --- NEW CHECKS ADDED HERE ---
def check_contrast(img):
    """Checks if the image has sufficient contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    print(f"    [DEBUG {get_timestamp()}] Contrast check (Std Dev): {contrast:.2f}")
    if contrast < 30:
        return f"Image has very low contrast"
    return "Contrast level acceptable"

def check_aspect_ratio(img):
    """Checks for extreme aspect ratios that would distort on resize."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        print(f"    [WARN {get_timestamp()}] Invalid image dimensions received: {h}x{w}")
        return "Invalid image dimensions"
    aspect_ratio = w / h
    print(f"    [DEBUG {get_timestamp()}] Aspect ratio check: {aspect_ratio:.2f}")
    if aspect_ratio > 3.0 or aspect_ratio < (1/3.0):
        return f"Warning: Extreme aspect ratio may cause distortion"
    return "Aspect ratio acceptable"
# --- END OF NEW CHECKS ---


def preprocess_and_validate(img):
    print(f"  [INFO {get_timestamp()}] Starting preprocessing and validation...")
    results = []

    # New: Aspect Ratio check on the original image
    results.append(check_aspect_ratio(img))

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
    
    # New: Contrast check on the resized image
    results.append(check_contrast(img_resized))

    print(f"  [INFO {get_timestamp()}] Preprocessing and validation complete.")
    return img_resized, results
#=====================================================

# ====== Health Check ======
@app.route('/health', methods=['GET'])
def health():
    print(f"[INFO {get_timestamp()}] Health check endpoint was hit.")
    return jsonify({"status": "ok"}), 200

# ====== Load Emotion Model =====
MODEL_PATH = "facial_emotion_detection_model.keras"
print(f"[SETUP {get_timestamp()}] Loading emotion model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"[FATAL {get_timestamp()}] Emotion model file not found at: {MODEL_PATH}")
    # In a real app, you might want to exit here
model = load_model(MODEL_PATH)
print(f"[SETUP {get_timestamp()}] Emotion model loaded successfully")

class_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
print(f"[SETUP {get_timestamp()}] Emotion Class labels: {class_labels}")

# ====== Load DeepLab Human Segmentation Model =====
print(f"[SETUP {get_timestamp()}] Loading DeepLabV3 model for human detection...")
deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
print(f"[SETUP {get_timestamp()}] DeepLabV3 model loaded successfully")

# ====== Utility for Emotion Prediction =====
def predict_emotion(img):
    print(f"  [INFO {get_timestamp()}] Starting emotion prediction...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=(0, -1))
    print(f"    [DEBUG {get_timestamp()}] Emotion model input tensor shape: {input_tensor.shape}")
    
    preds = model.predict(input_tensor)
    print(f"    [DEBUG {get_timestamp()}] Raw emotion predictions: {preds[0]}")
    
    class_idx = np.argmax(preds[0])
    predicted_label = class_labels[class_idx]
    print(f"  [INFO {get_timestamp()}] Emotion prediction complete. Result: {predicted_label}")
    return predicted_label

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
    print(f"  [INFO {get_timestamp()}] Starting person detection...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess_for_deeplab(img_rgb)
    print(f"    [DEBUG {get_timestamp()}] DeepLab input tensor shape: {input_tensor.shape}")

    with torch.no_grad():
        output = deeplab(input_tensor)

    mask = get_person_mask(output)
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        print(f"    [DEBUG {get_timestamp()}] Largest contour BBox: [x={x}, y={y}, w={w}, h={h}]")
        feedback = get_alignment_feedback((x, y, w, h), img.shape)
        print(f"  [INFO {get_timestamp()}] Person detected. Feedback generated.")
        return True, feedback
    else:
        print(f"  [INFO {get_timestamp()}] No person detected.")
        return False, ["No person detected"]

# ====== Flask Routes ======

@app.route('/predict', methods=['POST'])
def predict():
    print(f"\n\n===== [REQUEST {get_timestamp()}] /predict Endpoint Hit =====")
    if 'file' not in request.files:
        print(f"  [ERROR {get_timestamp()}] No file provided in request.")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    print(f"  [INFO {get_timestamp()}] Received file: {file.filename}")
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print(f"  [ERROR {get_timestamp()}] Failed to decode image.")
        return jsonify({'error': 'Invalid image'}), 400
    
    print(f"  [INFO {get_timestamp()}] Image decoded successfully. Original shape: {img.shape}")

    # Run pre-checks
    img_resized, precheck_feedback = preprocess_and_validate(img)

    # Log pre-checks in terminal
    print("\n  --- Pre-check Results ---")
    for msg in precheck_feedback:
        print(f"    - {msg}")
    print("  -------------------------\n")


    # Predict emotion
    mood = predict_emotion(img_resized)
    print(f"  [RESULT {get_timestamp()}] Final Predicted Mood: {mood}")
    print(f"===== [REQUEST END {get_timestamp()}] /predict =====\n")

    return jsonify({
        'mood': mood,
        'precheck_feedback': precheck_feedback
    })


@app.route('/check_person', methods=['POST'])
def check_person():
    print(f"\n\n===== [REQUEST {get_timestamp()}] /check_person Endpoint Hit =====")
    if 'file' not in request.files:
        print(f"  [ERROR {get_timestamp()}] No file provided in request.")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    print(f"  [INFO {get_timestamp()}] Received file: {file.filename}")
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print(f"  [ERROR {get_timestamp()}] Failed to decode image.")
        return jsonify({'error': 'Invalid image'}), 400
    
    print(f"  [INFO {get_timestamp()}] Image decoded successfully. Original shape: {img.shape}")

    # Run pre-checks
    img_resized, precheck_feedback = preprocess_and_validate(img)

    # Human detection
    person_present, feedback = detect_person_and_feedback(img_resized)

    # Log in terminal
    print("\n  --- Pre-check Results ---")
    for msg in precheck_feedback:
        print(f"    - {msg}")
    print("  -------------------------\n")


    if person_present:
        print(f"  [RESULT {get_timestamp()}] Person detected: YES")
    else:
        print(f"  [RESULT {get_timestamp()}] Person detected: NO")

    print("\n  --- Alignment Feedback ---")
    for msg in feedback:
        print(f"    - {msg}")
    print("  --------------------------\n")

    print(f"===== [REQUEST END {get_timestamp()}] /check_person =====\n")

    return jsonify({
        'person_detected': person_present,
        'feedback': feedback,
        'precheck_feedback': precheck_feedback
    })


if __name__ == '__main__':
    print(f"[SETUP {get_timestamp()}] Starting Flask server...")
    # Using debug=False for production, but True is fine for development
    app.run(host='0.0.0.0', port=5000, debug=True)